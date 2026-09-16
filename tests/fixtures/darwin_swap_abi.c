#include <mach/mach.h>
#include <mach/host_info.h>
#include <mach/vm_statistics.h>
#include <stddef.h>
#include <stdio.h>
int main(void) {
    printf("{\"rev1_count\":%u,\"rev1_size\":%zu,\"swapins_offset\":%zu,\"swapouts_offset\":%zu}\n",
        HOST_VM_INFO64_REV1_COUNT,offsetof(vm_statistics64_data_t,swapped_count),
        offsetof(vm_statistics64_data_t,swapins),offsetof(vm_statistics64_data_t,swapouts));
    return 0;
}
