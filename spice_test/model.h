#pragma once

#include <spice/models/brunel.h>
#include <spice/models/brunel_with_plasticity.h>
#include <spice/models/vogels_abbott.h>


using Models = ::testing::Types<spice::vogels_abbott, spice::brunel, spice::brunel_with_plasticity>;

#define TEST_ALL_MODELS( X )   \
	template <typename T>      \
	struct X : ::testing::Test \
	{                          \
	};                         \
	TYPED_TEST_CASE( X, Models );