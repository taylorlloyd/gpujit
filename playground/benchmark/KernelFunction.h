#include "llvm/Pass.h"

#include "llvm/IR/Module.h"

#include <cuda.h>
#include <nvToolsExt.h>
#include <string>
#include <vector>
#include <map>

#include "Assumption.h"


class KernelFunction {
  private:
    static bool doneLLVMInit;
    static bool doneCUDAInit;
    static bool compiling;
    static llvm::PassRegistry* Registry;
    static llvm::LLVMContext Context;
    std::unique_ptr<llvm::Module> module;
    std::string fnName;
    AssumptionList allAssumptions;
    CUModuleMap cumodules;

  public:
    KernelFunction(void* bitcode, size_t len);
    KernelFunction(void* bitcode, size_t len, std::string fnName);
    const llvm::Module& getModule();
    CUfunction getCUFunction(const CUmodule&);
    CUresult launchKernel(int gridX, int gridY, int gridZ,
                          int blockX, int blockY, int blockZ,
                          int smem, CUstream stream, void** params);
    std::string getKernelName();
    ~KernelFunction();

  private:
    static std::string* moduleToPTX(llvm::Module &M);
    static CUmodule loadCUmodule(const std::string& ptx);
    static CUmodule compileModule(const AssumptionList&, const llvm::Module*);
    static void LLVMInit();
    static void CUDAInit();
    static void *compileModuleAsync_thread(void *);
    void compileModuleAsync(AssumptionList);
    void proposeAssumptions(int gridX, int gridY, int gridZ,
                            int blockX, int blockY, int blockZ,
                            int smem, void** params);
    void compileLikelyModule();
    bool hasAssumption(const Assumption& a);
    bool hasCompiledAssumptions(const AssumptionList&) const;
};
