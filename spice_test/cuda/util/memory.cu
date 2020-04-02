__global__ void write( int * p ) { *p = 23; }

void write_23( int * p ) { write<<<1, 1>>>( p ); }