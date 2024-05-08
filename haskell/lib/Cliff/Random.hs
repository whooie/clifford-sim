-- | Partial functions for RNG, because I don't want to immediately get booted
-- into monadic operations just for that.
module Cliff.Random
  ( newRng
  , newRngSeed
  , genf
  , genfMulti
  , genb
  , genbMulti
  ) where

import Control.Monad (replicateM)
import Data.Function ((&))
import System.Random.Stateful (StatefulGen, StdGen, mkStdGen, uniformRM)

rngSeed :: Int
rngSeed = 10546

-- | Default RNG, seeded with @10546@.
newRng :: StdGen
newRng = mkStdGen rngSeed

-- | Default seedable RNG, with seed defaulting to @10546@.
newRngSeed :: Maybe Int -> StdGen
newRngSeed maybeSeed =
  case maybeSeed of
    Just n -> mkStdGen n
    Nothing -> newRng

-- | Shorthand for generating a single @Float@ in the [0.0, 1.0] range.
genf :: StatefulGen r m => r -> m Float
genf rng = uniformRM (0.0, 1.0) rng

-- | Generate a single @Bool@.
genb :: StatefulGen r Maybe => r -> Bool
genb rng =
  case uniformRM (False, True) rng of
    Just b -> b
    Nothing -> error "bad rng in genb"

-- | Generate a series of @Float@s.
genfMulti :: StatefulGen r Maybe => r -> Int -> [Float]
genfMulti rng n =
  case uniformRM (0.0, 1.0) rng & replicateM n of
    Just ff -> ff
    Nothing -> error "bad rng in genfMulti"

-- | Generate a series of @Bool@s.
genbMulti :: StatefulGen r Maybe => r -> Int -> [Bool]
genbMulti rng n =
  case uniformRM (False, True) rng & replicateM n of
    Just bb -> bb
    Nothing -> error "bad rng in genbMulti"

