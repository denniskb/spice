#pragma once

#include <spice/util/host_defines.h>


namespace spice
{
namespace util
{
template <typename T>
class span2d
{
public:
	HYBRID span2d( T * data = nullptr, int width = 0 )
	    : _data( data )
	    , _width( width )
	{
	}

	HYBRID unsigned width() const { return _width; }

	HYBRID T * data() { return _data; }
	HYBRID T const * data() const { return _data; }

	HYBRID T * row( unsigned i ) { return &_data[i * _width]; }
	HYBRID T const * row( unsigned i ) const { return &_data[i * _width]; }

	// i = row, j = col
	HYBRID T & operator()( unsigned i, unsigned j ) { return row( i )[j]; }
	HYBRID T const & operator()( unsigned i, unsigned j ) const { return row( i )[j]; }

private:
	T * _data = nullptr;
	unsigned _width = 0;
};
} // namespace util
} // namespace spice