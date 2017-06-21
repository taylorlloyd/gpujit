#include "llvm/Pass.h"

#include "llvm/IR/Module.h"

#include <cuda.h>
#include <string>

class KernelFunction {
  private:
    static bool doneLLVMInit;
    static bool doneCUDAInit;
    static llvm::PassRegistry* Registry;
    static llvm::LLVMContext Context;
    std::unique_ptr<llvm::Module> module;
    CUmodule* cumodule = nullptr;
    std::string* ptx = nullptr;
    std::string fnName;
  public:
    KernelFunction(void* bitcode, size_t len);
    KernelFunction(void* bitcode, size_t len, std::string fnName);
    const llvm::Module& getModule();
    const std::string& getPTX();
    const CUmodule& getCUModule();
    CUfunction getCUFunction();
    CUresult launchKernel(int gridX, int gridY, int gridZ,
                          int blockX, int blockY, int blockZ,
                          int smem, CUstream stream, void** params);
    std::string getKernelName();
    ~KernelFunction();

  private:
    std::string* moduleToPTX(llvm::Module &M);
    CUmodule loadCUmodule(const std::string& ptx);

    void LLVMInit();
    void CUDAInit();
};
