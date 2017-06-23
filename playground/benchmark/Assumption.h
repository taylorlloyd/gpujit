#include "llvm/IR/Module.h"

#include <cuda.h>
/*
 * Models an assumption made when JIT-compiling a KernelFunction
 */
class Assumption {
  public:
    enum AsmpKind {AK_Geometry};
  private:
    int held=0;
    AsmpKind kind;
  public:
  enum Prediction {Never, Unlikely, Unknown, Likely, VeryLikely, Always};
  Assumption(AsmpKind ak) : kind(ak) {}
  Prediction willHold() const;
  /*
   * Given a particular Kernel invocation, returns whether or not the assumption holds for this invocation
   */
  void update_assumption(int gridX, int gridY, int gridZ, int blockX, int blockY, int blockZ, int smem, void** params);
  /*
   * Apply knowledge gained from this assumption to an LLVM module
   */
  virtual bool apply(llvm::Module* M) const;
  /*
   * Virtual method implementing equality
   */
  virtual bool equals(const Assumption& other) const;
  bool operator==(const Assumption& other) const;
  const AsmpKind& getKind() const {return kind;}
  virtual bool holds(int gridX, int gridY, int gridZ, int blockX, int blockY, int blockZ, int smem, void** params) const;
};

typedef std::vector<std::shared_ptr<Assumption>> AssumptionList;

struct cmpBySize {
  bool operator()(const AssumptionList& a, const AssumptionList& b) const {
    return a.size() > b.size();
  }
};

typedef std::map<AssumptionList,CUmodule,cmpBySize> CUModuleMap;

class GeometryAssumption : public Assumption {
  public:
    enum Dim {GridX, GridY, GridZ, BlockX, BlockY, BlockZ};
  private:
    Dim dim;
    int value;
  public:
    GeometryAssumption(Dim dim, int value) : Assumption(AK_Geometry), dim(dim), value(value) {}
    bool holds(int gridX, int gridY, int gridZ, int blockX, int blockY, int blockZ, int smem, void** params) const;
    bool apply(llvm::Module* M) const;
    bool equals(const Assumption& other) const;
  private:
    static std::string intrinsic_names[6];
  public:
    static bool classof(const Assumption* a) {
      return a->getKind() == AK_Geometry;
    }
};
