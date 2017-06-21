#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "KernelFunction.h"

#include <iostream>

using namespace llvm;

void KernelFunction::CUDAInit() {
    cuInit(0);
    int devices;
    CUresult err = cuDeviceGetCount(&devices);
    if(err != CUDA_SUCCESS || devices == 0) {
      errs() << "Error retrieving valid CUDA device\n";
    }
    CUdevice d;
    err = cuDeviceGet(&d, 0);
    if(err != CUDA_SUCCESS) {
      errs() << "Error retrieving default CUDA device\n";
    }
    CUcontext ctx;
    err = cuDevicePrimaryCtxRetain(&ctx, d);
    if(err != CUDA_SUCCESS) {
      errs() << "Error obtaining default CUDA context\n";
    }
    cuCtxPushCurrent(ctx);
    if(err != CUDA_SUCCESS) {
      errs() << "Error setting CUDA context for thread\n";
    }
    doneCUDAInit = true;
}

void KernelFunction::LLVMInit() {
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  PassRegistry *Registry = PassRegistry::getPassRegistry();
  initializeCore(*Registry);
  initializeCodeGen(*Registry);
  initializeLoopStrengthReducePass(*Registry);
  initializeLowerIntrinsicsPass(*Registry);
  initializeCountingFunctionInserterPass(*Registry);
  initializeUnreachableBlockElimLegacyPassPass(*Registry);
  initializeConstantHoistingLegacyPassPass(*Registry);
  initializeScalarOpts(*Registry);
  initializeVectorization(*Registry);
  initializeScalarizeMaskedMemIntrinPass(*Registry);
  initializeExpandReductionsPass(*Registry);
  initializeScavengerTestPass(*Registry);
  doneLLVMInit = true;
}

std::string* KernelFunction::moduleToPTX(Module &M) {
  if(!doneLLVMInit)
      LLVMInit();

  SMDiagnostic Err;

  Triple TheTriple = Triple(M.getTargetTriple());
  if (TheTriple.getTriple().empty())
    TheTriple.setTriple(sys::getDefaultTargetTriple());

  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget("", TheTriple,
                                                         Error);
  if (!TheTarget) {
    errs() << Error;
    return nullptr;
  }

  std::string CPUStr = "", FeaturesStr = "";
  CodeGenOpt::Level OLvl = CodeGenOpt::Default;

  TargetOptions Options;
  Options.MCOptions.ShowMCEncoding = false;
  Options.MCOptions.MCUseDwarfDirectory = false;
  Options.MCOptions.AsmVerbose = false;
  Options.MCOptions.PreserveAsmComments = false;
  //Options.MCOptions.IASSearchPaths = IncludeDirs;
  //Options.MCOptions.SplitDwarfFile = false;

  std::unique_ptr<TargetMachine> Target( TheTarget->createTargetMachine(TheTriple.getTriple(), CPUStr, FeaturesStr, Options, Reloc::Model::Static, CodeModel::Default, OLvl));

  assert(Target && "Could not allocate target machine!");

  // If we don't have a module then just exit now. We do this down
  // here since the CPU/Feature help is underneath the target machine
  // creation.


  // Build up all of the passes that we want to do to the module.
  legacy::PassManager PM;

  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));
  PM.add(new TargetLibraryInfoWrapperPass(TLII));

  // Add the target data from the target machine, if it exists, or the module.
  M.setDataLayout(Target->createDataLayout());
  SmallVector<char, 0> Buffer;
  raw_svector_ostream BOS(Buffer);

  if (Target->addPassesToEmitFile(PM, BOS, TargetMachine::CodeGenFileType::CGFT_AssemblyFile, true, 0, 0, 0, 0)) {
    errs() << "target does not support generation of this"
      << " file type!\n";
    return nullptr;
  }

  PM.run(M);

  return new std::string(Buffer.begin(),Buffer.end());
}

const Module& KernelFunction::getModule() {
    assert(module != nullptr);
    return *module;
}
const std::string& KernelFunction::getPTX() {
    if(ptx == nullptr)
        ptx = moduleToPTX((llvm::Module&) getModule());
    assert(ptx != nullptr);
    return *ptx;
}

KernelFunction::KernelFunction(void *bitcode, size_t len) {
    auto ir_buffer = MemoryBuffer::getMemBuffer(StringRef((char*)bitcode, len), "<internal>", false);
    SMDiagnostic error;
    auto module = parseIR(MemoryBufferRef(*ir_buffer), error, Context);
    if(!module)
    {
        std::string what;
        llvm::raw_string_ostream os(what);
        error.print("Error building KernelFunction", os);
        std::cerr << what;
    }
    this->module = std::move(module);
}

std::string KernelFunction::getKernelName() {
    const Module& M = getModule();
    auto nvvmAnnot = M.getNamedMetadata("nvvm.annotations");
    for(auto a = nvvmAnnot->op_begin(),e = nvvmAnnot->op_end(); a!=e; ++a) {
      if((*a)->getNumOperands() == 3) {
        if(auto t = dyn_cast<MDString>((*a)->getOperand(1))) {
          if(t->getString() == "kernel") {
            auto v = dyn_cast<ValueAsMetadata>((*a)->getOperand(0));
            assert(v && "Kernel is value");
            auto kf = dyn_cast<Function>(v->getValue());
            assert(kf && "Kernel is a function");
            return kf->getName();
          }
        }
      }
    }
    return "";
}

KernelFunction::~KernelFunction() {
    if(cumodule != nullptr)
        delete cumodule;
    if(ptx != nullptr)
        delete ptx;
}

const CUmodule& KernelFunction::getCUModule() {
    if(!cumodule) {
        cumodule = new CUmodule;
        *cumodule = loadCUmodule(getPTX());
    }
    return *cumodule;
}

CUfunction KernelFunction::getCUFunction() {
    CUfunction func;
    CUresult err = cuModuleGetFunction(&func, getCUModule(), getKernelName().c_str());
    if(err != CUDA_SUCCESS) {
        errs() << "Error loading function from CUmodule\n";
    }
    return func;
}

CUmodule KernelFunction::loadCUmodule(const std::string& ptx) {
    if(!doneCUDAInit)
        CUDAInit();
    CUmodule mod;
    CUresult err = cuModuleLoadData(&mod, ptx.c_str());
    if(err != CUDA_SUCCESS) {
        errs() << "Error loading PTX module into CUDA\n";
    }
    return mod;
}
CUresult KernelFunction::launchKernel(
                      int gridX, int gridY, int gridZ,
                      int blockX, int blockY, int blockZ,
                      int smem, CUstream stream, void** params) {
    return cuLaunchKernel(getCUFunction(), gridX, gridY, gridZ, blockX, blockY, blockZ, smem, stream, params);
}

bool KernelFunction::doneLLVMInit = false;
bool KernelFunction::doneCUDAInit = false;
llvm::LLVMContext KernelFunction::Context;
