#include <gtest/gtest.h>

#include <spice/util/type_traits.h>


using namespace spice::util;


TEST( Int, IntCast )
{
	// clang-format off

	// SIGNED -> SIGNED
	// same
	ASSERT_EQ(  37, narrow_int<int>(  37));
	ASSERT_EQ(   1, narrow_int<int>(   1));
	ASSERT_EQ(   0, narrow_int<int>(   0));
	ASSERT_EQ(-  1, narrow_int<int>(-  1));
	ASSERT_EQ(-256, narrow_int<int>(-256));
	
	// small -> large
	ASSERT_EQ( 127, narrow_int<int>((char) 127));
	ASSERT_EQ(   1, narrow_int<int>((char)   1));
	ASSERT_EQ(   0, narrow_int<int>((char)   0));
	ASSERT_EQ(-  1, narrow_int<int>((char)-  1));
	ASSERT_EQ(-128, narrow_int<int>((char)-128));

	// large -> small
	ASSERT_THROW(             narrow_int<char>( 200), std::bad_cast);
	ASSERT_THROW(             narrow_int<char>( 128), std::bad_cast);
	ASSERT_EQ   ((char)  127, narrow_int<char>( 127));
	ASSERT_EQ   ((char)    1, narrow_int<char>(   1));
	ASSERT_EQ   ((char)    0, narrow_int<char>(   0));
	ASSERT_EQ   ((char) -  1, narrow_int<char>(-  1));
	ASSERT_EQ   ((char) -128, narrow_int<char>(-128));
	ASSERT_THROW(             narrow_int<char>(-129), std::bad_cast);
	ASSERT_THROW(             narrow_int<char>(-300), std::bad_cast);

	// UNSIGNED -> UNSIGNED
	// same
	ASSERT_EQ(400u, narrow_int<unsigned>(400u));
	ASSERT_EQ(  1u, narrow_int<unsigned>(  1u));
	ASSERT_EQ(  0u, narrow_int<unsigned>(  0u));

	// small -> large
	ASSERT_EQ(255u, narrow_int<unsigned>((unsigned char) 255));
	ASSERT_EQ(  1u, narrow_int<unsigned>((unsigned char)   1));
	ASSERT_EQ(  0u, narrow_int<unsigned>((unsigned char)   0));

	// large -> small
	ASSERT_THROW(                     narrow_int<unsigned char>(512u), std::bad_cast);
	ASSERT_THROW(                     narrow_int<unsigned char>(256u), std::bad_cast);
	ASSERT_EQ   ((unsigned char) 255, narrow_int<unsigned char>(255u));
	ASSERT_EQ   ((unsigned char)   1, narrow_int<unsigned char>(  1u));
	ASSERT_EQ   ((unsigned char)   0, narrow_int<unsigned char>(  0u));

	// SIGNED -> UNSIGNED
	// same
	ASSERT_EQ   (127u, narrow_int<unsigned>( 127));
	ASSERT_EQ   (  1u, narrow_int<unsigned>(   1));
	ASSERT_EQ   (  0u, narrow_int<unsigned>(   0));
	ASSERT_THROW(      narrow_int<unsigned>(-  1), std::bad_cast);
	ASSERT_THROW(      narrow_int<unsigned>(-100), std::bad_cast);

	// small -> large
	ASSERT_EQ   (127u, narrow_int<unsigned>((char)  127));
	ASSERT_EQ   (  1u, narrow_int<unsigned>((char)    1));
	ASSERT_EQ   (  0u, narrow_int<unsigned>((char)    0));
	ASSERT_THROW(      narrow_int<unsigned>((char) -  1), std::bad_cast);
	ASSERT_THROW(      narrow_int<unsigned>((char) - 55), std::bad_cast);

	// large -> small
	ASSERT_THROW(                     narrow_int<unsigned char>( 700), std::bad_cast);
	ASSERT_THROW(                     narrow_int<unsigned char>( 256), std::bad_cast);
	ASSERT_EQ   ((unsigned char) 255, narrow_int<unsigned char>( 255));
	ASSERT_EQ   ((unsigned char)   1, narrow_int<unsigned char>(   1));
	ASSERT_EQ   ((unsigned char)   0, narrow_int<unsigned char>(   0));
	ASSERT_THROW(                     narrow_int<unsigned char>(-  1), std::bad_cast);
	ASSERT_THROW(                     narrow_int<unsigned char>(-100), std::bad_cast);

	// UNSIGNED -> SIGNED
	// same
	ASSERT_THROW(            narrow_int<int>(4000000000u), std::bad_cast);
	ASSERT_THROW(            narrow_int<int>(2147483648u), std::bad_cast);
	ASSERT_EQ   (2147483647, narrow_int<int>(2147483647u));
	ASSERT_EQ   (         1, narrow_int<int>(         1u));
	ASSERT_EQ   (         0, narrow_int<int>(         0u));

	// small -> large
	// (can't generate a *small* unsigned that, when converted to a *big* singed, throws.)
	ASSERT_EQ(127, narrow_int<int>((unsigned char) 127));
	ASSERT_EQ(  1, narrow_int<int>((unsigned char)   1));
	ASSERT_EQ(  0, narrow_int<int>((unsigned char)   0));

	// large -> small
	ASSERT_THROW(           narrow_int<char>(200u), std::bad_cast);
	ASSERT_THROW(           narrow_int<char>(128u), std::bad_cast);
	ASSERT_EQ(  (char) 127, narrow_int<char>(127u));
	ASSERT_EQ(  (char)   1, narrow_int<char>(  1u));
	ASSERT_EQ(  (char)   0, narrow_int<char>(  0u));

	// clang-format on
}