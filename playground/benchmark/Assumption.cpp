#include "Assumption.h"
#include "llvm/IR/Instructions.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
/***************************************
 * Assumption
 **************************************/

void Assumption::update_assumption(int gridX, int gridY, int gridZ, int blockX, int blockY, int blockZ, int smem, void** params) {
    bool dispatch = holds(gridX, gridY, gridZ, blockX, blockY, blockZ, smem, params);
    held = held << 1 | (dispatch ? 1 : 0);
    errs() << "Assumption likelihood: (" << held << ") " << willHold() << "\n";
}

Assumption::Prediction Assumption::willHold() const {
    // Standard 2-bit saturating counter, with 3-bit VLikely
    if((held & 0x03) == 0)
        return Prediction::Unlikely;
    else if((held & 0x07) == 7)
        return Prediction::VeryLikely;
    else if((held & 0x03) == 3)
        return Prediction::Likely;
    else
        return Prediction::Unknown;
}

bool Assumption::holds(int gridX, int gridY, int gridZ, int blockX, int blockY, int blockZ, int smem, void** params) const {
    return true;
}
bool Assumption::apply(llvm::Module* M) const {
    return true;
}
bool Assumption::operator==(const Assumption& other) const {
    return equals(other);
}
bool Assumption::equals(const Assumption& other) const {
    return false;
}

/***************************************
 * GeometryAssumption
 **************************************/

std::string GeometryAssumption::intrinsic_names[] = {
  "llvm.nvvm.read.ptx.sreg.nctaid.x",
  "llvm.nvvm.read.ptx.sreg.nctaid.y",
  "llvm.nvvm.read.ptx.sreg.nctaid.z",
  "llvm.nvvm.read.ptx.sreg.ntid.x",
  "llvm.nvvm.read.ptx.sreg.ntid.y",
  "llvm.nvvm.read.ptx.sreg.ntid.z"
};

bool GeometryAssumption::holds(int gridX, int gridY, int gridZ, int blockX, int blockY, int blockZ, int smem, void** params) const {
    return (dim == Dim::GridX && value == gridX) ||
           (dim == Dim::GridY && value == gridY) ||
           (dim == Dim::GridZ && value == gridZ) ||
           (dim == Dim::BlockX && value == blockX) ||
           (dim == Dim::BlockY && value == blockY) ||
           (dim == Dim::BlockZ && value == blockZ);
}
bool GeometryAssumption::apply(llvm::Module* M) const {
    using namespace llvm;
    for(auto F=M->begin(),e=M->end(); F!=e; ++F) {
      for(auto B=F->begin(),e=F->end(); B!=e; ++B) {
        bool changed = true;
        while(changed) {
          changed = false;
          for(auto I=B->begin(),e=B->end(); I!=e; ++I) {
            // If this is the appropriate intrinsic,
            if(auto call=dyn_cast<CallInst>(I)) {
              errs() << call->getCalledFunction()->getName() << "\n";
            }
            // replace it with a constant of our value
          }
        }
      }
    }
    return true;
}

bool GeometryAssumption::equals(const Assumption& a) const {
    if(auto ga = dyn_cast<GeometryAssumption>(&a)) {
        return ga->dim == dim && ga->value == value;
    }
    return false;
}

