-- | Definitions for simple RNG.
module Cliff.Random
  ( R
  , runR
  , randBool
  ) where

import Control.Monad.State
import System.Random.Stateful

-- | Denotes a randomized "action" outputting a type @a@ to be realized by
-- `runR`.
type R a = State StdGen a

-- | Runner for a randomized action (@R a@), given a seed.
--
-- > -- this program defines a lazy, infinite stream of random Floats
-- >
-- > import System.Random.Stateful
-- >
-- > randFloat :: R Float
-- > randFloat = do
-- >   gen <- get
-- >   let (f, gen') = random gen
-- >   put gen'
-- >   return f
-- >
-- > floatStream :: R [Float]
-- > floatStream = mapM (\_ -> randFloat) $ repeat ()
-- >
-- > getFloats :: Int -> [Float]
-- > getFloats seed = runR seed floatStream
runR
  -- | RNG seed
  :: Int
  -- | Randomized action
  -> R a
  -- | Action output
  -> a
runR seed action = evalState action $ mkStdGen seed

-- | A random generator action for a single @Bool@.
randBool :: R Bool
randBool = do
  gen <- get
  let (r, gen') = random gen
  put gen'
  return r

