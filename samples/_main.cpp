#include "../spice/cuda/snn.h"
#include "../spice/models/brunel.h"
#include "../spice/models/brunel_with_plasticity.h"
#include "../spice/models/vogels_abbott.h"

#include <cstring>
#include <iostream>


using namespace spice;
using namespace spice::util;


static unsigned nn( int nsyn, double p ) { return narrow_cast<unsigned>( std::sqrt( nsyn / p ) ); }

// Usage: samples.exe {brunel|brunel+|vogels} [synapse-count] [iter-count]
int main( int const argc, char const ** argv )
{
	if( argc != 4 )
	{
		std::cout << "Usage: spice_bench.exe {brunel|brunel+|vogels} [synapse-count] [iter-count]"
		          << std::endl;
		return EXIT_FAILURE;
	}

	int const NSYN = atoi( argv[2] );
	int const NITER = atoi( argv[3] );

	// Lambda taking a snn and simulating it for NITER steps.
	// Neuron activations get written out to [out-file].
	auto run_sim = [=]( auto && net ) {
		std::cout << net.num_neurons() << std::endl;

		std::vector<int> n;
		for( std::size_t i = 0; i < NITER; i++ )
		{
			// Advance snn by a single simulation step, store activations in 'n'.
			net.step( &n );

			// Ommit simulation steps without activations from output.
			// if( n.size() > 0 )
			{
				std::sort( n.begin(), n.end() );

				auto delim = "";
				for( int x : n )
				{
					std::cout << delim << x;
					delim = ",";
				}
				std::cout << std::endl;
			}
		}
	};

	try
	{
		// Initialize a snn with the brunel model
		if( !strcmp( argv[1], "brunel" ) )
			run_sim( cuda::snn<brunel>(
			    { { nn( NSYN, 0.05 ) / 2, nn( NSYN, 0.05 ) / 2 },
			      { { 0, 1, 0.1f }, { 1, 1, 0.1f } } },
			    0.0001f,
			    15 ) );
		// Initialize a snn with the brunel model (with plasticity turned on)
		else if( !strcmp( argv[1], "brunel+" ) )
			run_sim( cuda::snn<brunel_with_plasticity>(
			    { { nn( NSYN, 0.05 ) / 2, nn( NSYN, 0.05 ) / 2 },
			      { { 0, 1, 0.1f }, { 1, 1, 0.1f } } },
			    0.0001f,
			    15 ) );
		// Initialize a snn with the vogels&abbott model
		else if( !strcmp( argv[1], "vogels" ) )
			run_sim( cuda::snn<vogels_abbott>( { nn( NSYN, 0.02 ), 0.02f }, 0.0001f, 8 ) );
	}
	catch( std::exception & e )
	{
		std::cout << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return 0;
}