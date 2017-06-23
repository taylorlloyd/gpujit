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
#include "llvm/Transforms/Utils/Cloning.h"
#include "KernelFunction.h"

#include <iostream>
#include <unistd.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/syscall.h>

using namespace llvm;

void KernelFunction::CUDAInit() {
    nvtxRangePush("CUDAInit");
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
    nvtxRangePop();
}

void KernelFunction::LLVMInit() {
  nvtxRangePush("LLVMInit");
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
  nvtxRangePop();
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
    this->fnName = "";
}

KernelFunction::KernelFunction(void *bitcode, size_t len, std::string fnName) {
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
    this->fnName = fnName;
}

std::string KernelFunction::getKernelName() {
    if(!fnName.empty())
        return fnName;
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

CUfunction KernelFunction::getCUFunction(const CUmodule& M) {
    CUfunction func;
    CUresult err = cuModuleGetFunction(&func, M, getKernelName().c_str());
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
    CUmodule mod;
    CUmodule* M = nullptr;
    for(auto t=cumodules.begin(),e=cumodules.end(); t!=e; ++t) {
      bool valid = true;
      for(auto a=t->first.begin(),e=t->first.end(); a!=e; ++a) {
        if(!(*a)->holds(gridX, gridY, gridZ, blockX, blockY, blockZ, smem, params)) {
          valid = false;
          break;
        }
      }
      if(valid) {
        M = &(t->second);
        break;
      }
    }
    if(M == nullptr) {
        // Generate the default module synchronously
        AssumptionList assumptions;
        mod = compileModule(assumptions, &getModule());
        cumodules.insert(make_pair(assumptions, mod));
        M = &mod;
    }
    // Propose Assumptions
    proposeAssumptions(gridX, gridY, gridZ, blockX, blockY, blockZ, smem, params);
    // Update Assumptions
    for(auto a=allAssumptions.begin(),e=allAssumptions.end(); a!=e; ++a) {
        (*a)->update_assumption(gridX, gridY, gridZ, blockX, blockY, blockZ, smem, params);
    }
    // Trigger possible recompilation
    compileLikelyModule();

    return cuLaunchKernel(getCUFunction(*M), gridX, gridY, gridZ, blockX, blockY, blockZ, smem, stream, params, NULL);
}

CUmodule KernelFunction::compileModule(const AssumptionList& assumptions, const llvm::Module* orig_module) {
    compiling = true;
    nvtxRangePush("compileModule");
    // Make our own copy of the module
    std::unique_ptr<llvm::Module> M = CloneModule(orig_module);

    // Apply any assumptions
    nvtxRangePush("JIT Optimizations");
    for(auto a=assumptions.begin(),e=assumptions.end(); a!=e; ++a) {
        (*a)->apply(&*M);
    }
    nvtxRangePop();

    // Run compilation flow
    nvtxRangePush("LLVM to PTX");
    std::string* ptx = moduleToPTX(*M);
    nvtxRangePop();
    assert(ptx != nullptr);
    nvtxRangePush("PTX to SASS");
    CUmodule cumod = loadCUmodule(*ptx);
    nvtxRangePop();
    delete ptx;
    nvtxRangePop();
    compiling = false;
    return cumod;
}

struct compileModule_args {
    AssumptionList assumptions;
    const llvm::Module* module;
    CUModuleMap* cumodules;
    CUcontext ctx;
};

void *KernelFunction::compileModuleAsync_thread(void *v_args) {
    // Thread init stuff
    pid_t tid = syscall(SYS_gettid);
    nvtxNameOsThread(tid, "BackgroundCompile");
    errs() << "KernelFunction: beginning background compilation.\n";

    // Extract our arguments
    struct compileModule_args* args = (struct compileModule_args*) v_args;
    cuCtxPushCurrent(args->ctx);
    // Perform the compilation
    CUmodule cumodule = compileModule(args->assumptions, args->module);
    // Save the result
    args->cumodules->insert(make_pair(args->assumptions, cumodule));
    // We're done with the arguments
    delete v_args;
    return nullptr;
}

void KernelFunction::compileModuleAsync(AssumptionList assumptions) {
    // Create the arguments
    struct compileModule_args* args = new compileModule_args;
    args->assumptions = assumptions;
    args->module = &getModule();
    args->cumodules = &cumodules;
    cuCtxGetCurrent(&args->ctx);
    pthread_t bg_thread;
    pthread_create(&bg_thread, NULL, KernelFunction::compileModuleAsync_thread, args);
}
void KernelFunction::proposeAssumptions(
                        int gridX, int gridY, int gridZ,
                        int blockX, int blockY, int blockZ,
                        int smem, void** params) {
    // Generate GeometryAssumptions
    auto assumeGX = std::make_shared<GeometryAssumption>(GeometryAssumption::GridX, gridX);
    auto assumeGY = std::make_shared<GeometryAssumption>(GeometryAssumption::GridY, gridY);
    auto assumeGZ = std::make_shared<GeometryAssumption>(GeometryAssumption::GridZ, gridZ);
    if(!hasAssumption(*assumeGX))
      allAssumptions.push_back(assumeGX);
    if(!hasAssumption(*assumeGY))
      allAssumptions.push_back(assumeGY);
    if(!hasAssumption(*assumeGZ))
      allAssumptions.push_back(assumeGZ);

    auto assumeBX = std::make_shared<GeometryAssumption>(GeometryAssumption::BlockX, blockX);
    auto assumeBY = std::make_shared<GeometryAssumption>(GeometryAssumption::BlockY, blockY);
    auto assumeBZ = std::make_shared<GeometryAssumption>(GeometryAssumption::BlockZ, blockZ);
    if(!hasAssumption(*assumeBX))
      allAssumptions.push_back(assumeBX);
    if(!hasAssumption(*assumeBY))
      allAssumptions.push_back(assumeBY);
    if(!hasAssumption(*assumeBZ))
      allAssumptions.push_back(assumeBZ);

    // TODO: Generate additional assumptions
    errs() << getKernelName() << ": Now has " << allAssumptions.size() << " possible assumptions.\n";
}
bool KernelFunction::hasAssumption(const Assumption& a) {
    for(auto b=allAssumptions.begin(),e=allAssumptions.end(); b!=e; ++b) {
      if(**b == a)
          return true;
    }
    return false;
}

void KernelFunction::compileLikelyModule() {
    // Collect the list of safe assumptions
    AssumptionList likely;
    for(auto a=allAssumptions.begin(),e=allAssumptions.end(); a!=e; ++a) {
        if((*a)->willHold() >= Assumption::Likely)
            likely.push_back(*a);
    }
    errs() << getKernelName() << ": " << likely.size() << " likely assumptions.\n";


    if(!hasCompiledAssumptions(likely) && !compiling) {
        // Let's build a new module!
        errs() << getKernelName() << ": Recompiling.\n";
        compileModuleAsync(likely);
    } else {
        errs() << getKernelName() << ": Likely assumptions match existing compilation.\n";
    }
}

bool KernelFunction::hasCompiledAssumptions(const AssumptionList& candidate) const {
    // Check if we have this particular list compiled
    // (This is terribly inefficient)
    bool exists = false;
    for(auto t=cumodules.begin(),e=cumodules.end(); t!=e; ++t) {
      if(t->first.size() == candidate.size()) {
          // potential match, ensure all our elements are contained
          bool elems_match = true;
          for(auto l=candidate.begin(),e=candidate.end(); l!=e && elems_match; ++l) {
            bool e_match = false;
            for(auto existing=t->first.begin(),e=t->first.end();existing!=e && !e_match; ++existing) {
              if(**l == **existing)
                  e_match = true;
            }
            elems_match &= e_match;
          }
          if (elems_match) {
            exists = true;
            break;
          }
      }
    }
    return exists;
}

bool KernelFunction::compiling = false;
bool KernelFunction::doneLLVMInit = false;
bool KernelFunction::doneCUDAInit = false;
llvm::LLVMContext KernelFunction::Context;
