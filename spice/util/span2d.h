#pragma once

#include <spice/util/host_defines.h>
#include <spice/util/stdint.h>


namespace spice
{
namespace util
{
template <typename T>
class span2d
{
public:
	HYBRID span2d( T * data = nullptr, int_ width = 0 )
	    : _data( data )
	    , _width( width )
	{
	}

	HYBRID uint_ width() const { return _width; }

	HYBRID T * data() { return _data; }
	HYBRID T const * data() const { return _data; }

	HYBRID T * row( uint_ i ) { return &_data[i * _width]; }
	HYBRID T const * row( uint_ i ) const { return &_data[i * _width]; }

	// i = row, j = col
	HYBRID T & operator()( uint_ i, uint_ j ) { return row( i )[j]; }
	HYBRID T const & operator()( uint_ i, uint_ j ) const { return row( i )[j]; }

private:
	T * _data = nullptr;
	uint_ _width = 0;
};
} // namespace util
} // namespace spice