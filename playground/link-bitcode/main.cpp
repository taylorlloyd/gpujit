#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

// These will be injected by objcopy
extern char _binary_kernel_bc_start;
extern char _binary_kernel_bc_end;

int main() {
    printf("Start: %lX, size: %ld bytes\n",(size_t) &_binary_kernel_bc_start, ((size_t)&_binary_kernel_bc_start)-((size_t)&_binary_kernel_bc_end));

    char* bitcode = &_binary_kernel_bc_start;
    size_t len = ((size_t)&_binary_kernel_bc_end)-((size_t)&_binary_kernel_bc_start);

    LLVMContext context;
    SMDiagnostic error;
    auto ir_buffer = MemoryBuffer::getMemBuffer(StringRef(bitcode, len), "<internal>", false);
    auto module = parseIR(MemoryBufferRef(*ir_buffer), error, context);

    if(!module)
    {
        std::string what;
        llvm::raw_string_ostream os(what);
        error.print("error after ParseIR()", os);
        std::cerr << what;
    }
    errs() << *module;
    return 0;
}
