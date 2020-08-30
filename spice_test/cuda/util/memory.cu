#include <spice/util/stdint.h>


__global__ void write( int_ * p ) { *p = 23; }

void write_23( int_ * p ) { write<<<1, 1>>>( p ); }
