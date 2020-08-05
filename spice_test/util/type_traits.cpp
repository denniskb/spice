#include <gtest/gtest.h>

#include <spice/util/type_traits.h>


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

	// UNSIGNED -> UNSIGNED
	// small -> large
	ASSERT_EQ(255u, narrow<unsigned>((unsigned char) 255));
	ASSERT_EQ(  1u, narrow<unsigned>((unsigned char)   1));
	ASSERT_EQ(  0u, narrow<unsigned>((unsigned char)   0));

	// large -> small
	ASSERT_THROW(                     narrow<unsigned char>(512u), std::bad_cast);
	ASSERT_THROW(                     narrow<unsigned char>(256u), std::bad_cast);
	ASSERT_EQ   ((unsigned char) 255, narrow<unsigned char>(255u));
	ASSERT_EQ   ((unsigned char)   1, narrow<unsigned char>(  1u));
	ASSERT_EQ   ((unsigned char)   0, narrow<unsigned char>(  0u));

	// SIGNED -> UNSIGNED
	// same
	ASSERT_EQ   (127u, narrow<unsigned>( 127));
	ASSERT_EQ   (  1u, narrow<unsigned>(   1));
	ASSERT_EQ   (  0u, narrow<unsigned>(   0));
	ASSERT_THROW(      narrow<unsigned>(-  1), std::bad_cast);
	ASSERT_THROW(      narrow<unsigned>(-100), std::bad_cast);

	// small -> large
	ASSERT_EQ   (127u, narrow<unsigned>((char)  127));
	ASSERT_EQ   (  1u, narrow<unsigned>((char)    1));
	ASSERT_EQ   (  0u, narrow<unsigned>((char)    0));
	ASSERT_THROW(      narrow<unsigned>((char) -  1), std::bad_cast);
	ASSERT_THROW(      narrow<unsigned>((char) - 55), std::bad_cast);

	// large -> small
	ASSERT_THROW(                     narrow<unsigned char>( 700), std::bad_cast);
	ASSERT_THROW(                     narrow<unsigned char>( 256), std::bad_cast);
	ASSERT_EQ   ((unsigned char) 255, narrow<unsigned char>( 255));
	ASSERT_EQ   ((unsigned char)   1, narrow<unsigned char>(   1));
	ASSERT_EQ   ((unsigned char)   0, narrow<unsigned char>(   0));
	ASSERT_THROW(                     narrow<unsigned char>(-  1), std::bad_cast);
	ASSERT_THROW(                     narrow<unsigned char>(-100), std::bad_cast);

	// UNSIGNED -> SIGNED
	// same
	ASSERT_THROW(            narrow<int>(4000000000u), std::bad_cast);
	ASSERT_THROW(            narrow<int>(2147483648u), std::bad_cast);
	ASSERT_EQ   (2147483647, narrow<int>(2147483647u));
	ASSERT_EQ   (         1, narrow<int>(         1u));
	ASSERT_EQ   (         0, narrow<int>(         0u));

	// small -> large
	// (can't generate a *small* unsigned that, when converted to a *big* singed, throws.)
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
	ASSERT_EQ(3.14f, narrow<double>(3.14f));
	ASSERT_EQ(0.1f, narrow<double>(0.1f));
	ASSERT_EQ(0.125f, narrow<float>(0.125));
	ASSERT_THROW(narrow<float>(0.1), std::bad_cast);

	// int -> real
	ASSERT_EQ(314.f, narrow<float>(314));
	ASSERT_EQ(314.0, narrow<double>(314));
	ASSERT_EQ(-123456.f, narrow<float>(-123456));
	ASSERT_THROW(narrow<float>(1234567890), std::bad_cast);
	ASSERT_EQ(0x1p30, narrow<float>(1<<30));
	ASSERT_THROW(narrow<double>(9007199254740993llu), std::bad_cast);
	ASSERT_EQ(0x1p60, narrow<double>(1llu << 60));
	ASSERT_THROW(narrow<float>(INT_MAX), std::bad_cast);
	ASSERT_EQ((float)INT_MIN, narrow<float>(INT_MIN));

	// real -> int
	ASSERT_EQ(3, narrow<int>(3.f));
	ASSERT_EQ(3, narrow<int>(3.0));
	ASSERT_THROW(narrow<int>(3.14f), std::bad_cast);
	ASSERT_THROW(narrow<int>(3.14), std::bad_cast);
	ASSERT_EQ(-7, narrow<int>(-7.f));
	ASSERT_THROW(narrow<unsigned>(-7.0), std::bad_cast);
	ASSERT_THROW(narrow<int>(0x1p31), std::bad_cast);
	ASSERT_EQ(1llu << 31, narrow<unsigned>(0x1p31));
	ASSERT_THROW(narrow<unsigned>(0x1p32), std::bad_cast);
	ASSERT_THROW(narrow<int>((float)INT_MAX), std::bad_cast);

	// clang-format on
}