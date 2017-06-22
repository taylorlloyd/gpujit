#include "llvm/IR/Module.h"

#include <cuda.h>
/*
 * Models an assumption made when JIT-compiling a KernelFunction
 */
class Assumption {
  public:
  /*
   * Given a particular Kernel invocation, returns whether or not the assumption holds for this invocation
   */
  virtual bool holds(int gridX, int gridY, int gridZ, int blockX, int blockY, int blockZ, int smem, void** params) const;
  /*
   * Apply knowledge gained from this assumption to an LLVM module
   */
  virtual bool apply(llvm::Module* M) const;
};

typedef std::vector<Assumption> AssumptionList;

struct cmpBySize {
  bool operator()(const AssumptionList& a, const AssumptionList& b) const {
    return a.size() > b.size();
  }
};

typedef std::map<AssumptionList,CUmodule,cmpBySize> CUModuleMap;

class GeometryAssumption : Assumption {
  public:
    enum Dim {GridX, GridY, GridZ, BlockX, BlockY, BlockZ};
  private:
    Dim dim;
    int value;
  public:
    GeometryAssumption(Dim dim, int value) : dim{dim}, value{value} {}
    bool holds(int gridX, int gridY, int gridZ, int blockX, int blockY, int blockZ, int smem, void** params) const;
    bool apply(llvm::Module* M) const;
  private:
    static std::string intrinsic_names[6];
};
