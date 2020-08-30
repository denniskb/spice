#include <gtest/gtest.h>

#include <spice/util/random.h>
#include <spice/util/stdint.h>
#include <spice/util/type_traits.h>

#include <cmath>
#include <cstring>


using namespace spice::util;


TEST( TypeTraits, Narrow )
{
	// clang-format off

	// SIGNED -> SIGNED	
	// small -> large
	ASSERT_EQ( 127, narrow<int>((char) 127));
	ASSERT_EQ(   1, narrow<int>((char)   1));
	ASSERT_EQ(   0, narrow<int>((char)   0));
	ASSERT_EQ(-  1, narrow<int>((char)-  1));
	ASSERT_EQ(-128, narrow<int>((char)-128));

	// large -> small
	ASSERT_THROW(             narrow<char>( 200), std::bad_cast);
	ASSERT_THROW(             narrow<char>( 128), std::bad_cast);
	ASSERT_EQ   ((char)  127, narrow<char>( 127));
	ASSERT_EQ   ((char)    1, narrow<char>(   1));
	ASSERT_EQ   ((char)    0, narrow<char>(   0));
	ASSERT_EQ   ((char) -  1, narrow<char>(-  1));
	ASSERT_EQ   ((char) -128, narrow<char>(-128));
	ASSERT_THROW(             narrow<char>(-129), std::bad_cast);
	ASSERT_THROW(             narrow<char>(-300), std::bad_cast);

	// uint_ -> uint_
	// small -> large
	ASSERT_EQ(255u, narrow<uint_>((unsigned char) 255));
	ASSERT_EQ(  1u, narrow<uint_>((unsigned char)   1));
	ASSERT_EQ(  0u, narrow<uint_>((unsigned char)   0));

	// large -> small
	ASSERT_THROW(                     narrow<unsigned char>(512u), std::bad_cast);
	ASSERT_THROW(                     narrow<unsigned char>(256u), std::bad_cast);
	ASSERT_EQ   ((unsigned char) 255, narrow<unsigned char>(255u));
	ASSERT_EQ   ((unsigned char)   1, narrow<unsigned char>(  1u));
	ASSERT_EQ   ((unsigned char)   0, narrow<unsigned char>(  0u));

	// SIGNED -> uint_
	// same
	ASSERT_EQ   (127u, narrow<uint_>( 127));
	ASSERT_EQ   (  1u, narrow<uint_>(   1));
	ASSERT_EQ   (  0u, narrow<uint_>(   0));
	ASSERT_THROW(      narrow<uint_>(-  1), std::bad_cast);
	ASSERT_THROW(      narrow<uint_>(-100), std::bad_cast);

	// small -> large
	ASSERT_EQ   (127u, narrow<uint_>((char)  127));
	ASSERT_EQ   (  1u, narrow<uint_>((char)    1));
	ASSERT_EQ   (  0u, narrow<uint_>((char)    0));
	ASSERT_THROW(      narrow<uint_>((char) -  1), std::bad_cast);
	ASSERT_THROW(      narrow<uint_>((char) - 55), std::bad_cast);

	// large -> small
	ASSERT_THROW(                     narrow<unsigned char>( 700), std::bad_cast);
	ASSERT_THROW(                     narrow<unsigned char>( 256), std::bad_cast);
	ASSERT_EQ   ((unsigned char) 255, narrow<unsigned char>( 255));
	ASSERT_EQ   ((unsigned char)   1, narrow<unsigned char>(   1));
	ASSERT_EQ   ((unsigned char)   0, narrow<unsigned char>(   0));
	ASSERT_THROW(                     narrow<unsigned char>(-  1), std::bad_cast);
	ASSERT_THROW(                     narrow<unsigned char>(-100), std::bad_cast);

	// uint_ -> SIGNED
	// same
	ASSERT_THROW(            narrow<int>(4000000000u), std::bad_cast);
	ASSERT_THROW(            narrow<int>(2147483648u), std::bad_cast);
	ASSERT_EQ   (2147483647, narrow<int>(2147483647u));
	ASSERT_EQ   (         1, narrow<int>(         1u));
	ASSERT_EQ   (         0, narrow<int>(         0u));

	// small -> large
	ASSERT_EQ(127, narrow<int>((unsigned char) 127));
	ASSERT_EQ(  1, narrow<int>((unsigned char)   1));
	ASSERT_EQ(  0, narrow<int>((unsigned char)   0));

	// large -> small
	ASSERT_THROW(           narrow<char>(200u), std::bad_cast);
	ASSERT_THROW(           narrow<char>(128u), std::bad_cast);
	ASSERT_EQ(  (char) 127, narrow<char>(127u));
	ASSERT_EQ(  (char)   1, narrow<char>(  1u));
	ASSERT_EQ(  (char)   0, narrow<char>(  0u));

	// real -> real
	// small -> large
	{
		xoroshiro256ss rng(1337);
		for (int_ i = 0; i < 1'000'000; i++)
		{
			uint_ x = rng() >> 32;
			float f;
			std::memcpy(&f, &x, sizeof(x));
			if (!std::isnan(f)) { ASSERT_EQ(static_cast<double>(f), narrow<double>(f)); }
		}
	}
	// large -> small
	{
		xoroshiro256ss rng(1337);
		for (int_ i = 0; i < 1'000'000; i++)
		{
			uint_ x = rng() >> 32;
			float f;
			std::memcpy(&f, &x, sizeof(x));
			if (!std::isnan(f)) { ASSERT_EQ(f, narrow<float>(static_cast<double>(f))); }
		}
		ASSERT_THROW(narrow<float>(   0.1), std::bad_cast);
		ASSERT_THROW(narrow<float>(  -0.1), std::bad_cast);
		ASSERT_THROW(narrow<float>( 1e100), std::bad_cast);
		ASSERT_THROW(narrow<float>(-1e100), std::bad_cast);
	}

	// int_ -> real
	{
		int_ i = 0;
		float f = 0;
		double d = 0;
		for (;i <= (1<<24); i++, f++, d++)
		{
			ASSERT_EQ( f, narrow<float> ( i));
			ASSERT_EQ(-f, narrow<float> (-i));
			ASSERT_EQ( d, narrow<double>( i));
			ASSERT_EQ(-d, narrow<double>(-i));
		}
		ASSERT_THROW(         narrow<float>((1<<24) + 1), std::bad_cast);
		ASSERT_EQ(16777218.f, narrow<float>(16777218));

		ASSERT_THROW(          narrow<float>(123456791), std::bad_cast);
		ASSERT_EQ(123456792.f, narrow<float>(123456792));
		ASSERT_THROW(          narrow<float>(123456793), std::bad_cast);

		ASSERT_EQ(0x1p59, narrow<float>(1llu << 59));
		ASSERT_EQ(0x1p59, narrow<double>(1llu << 59));

		ASSERT_THROW(narrow<float>(std::numeric_limits<int>::max()), std::bad_cast);
	}

	// real -> int_
	{
		xoroshiro256ss rng(1337);
		for (int_ i = 0; i < 100'000; i++)
		{
			uint_ x = rng() >> 32;
			float f;
			std::memcpy(&f, &x, sizeof(x));
			if (!std::isnan(f))
			{
				float tmp;
				float frac = std::modf(f, &tmp);
				if (frac != 0) { ASSERT_THROW(narrow<int>(f), std::bad_cast); }
			}

			double d = static_cast<double>(x);
			ASSERT_EQ(x, narrow<uint_>(d));
		}
	}

	// clang-format on
}