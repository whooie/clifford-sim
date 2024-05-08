-- | Definitions for base qubit objects.
module Cliff.Tree.Qubit
  ( Qubit (..)
  , qFlip
  , qH
  , qX
  , qY
  , qZ
  , qS
  , Basis (..)
  , basisOutcomes
  , basisPlus
  , basisMinus
  , basisIsElem
  , basisGen
  ) where

import System.Random.Stateful (StatefulGen, StdGen)
import Cliff.Gate
import Cliff.Random

-- | Base qubit object defined as one of the 6 cardinal directions on the Bloch
-- sphere.
--
-- The ∣+z⟩ state corresponds to ∣0⟩.
data Qubit = Xp | Xm | Yp | Ym | Zp | Zm deriving (Eq, Enum)

instance Show Qubit where
  show Xp = "+x"
  show Xm = "-x"
  show Yp = "+y"
  show Ym = "-y"
  show Zp = "+z"
  show Zm = "-z"

-- | Flip the direction of a single qubit.
qFlip :: Qubit -> Qubit
qFlip Xp = Xm
qFlip Xm = Xp
qFlip Yp = Ym
qFlip Ym = Yp
qFlip Zp = Zm
qFlip Zm = Zp

-- | Apply a Hadamard gate to a qubit.
qH :: Qubit -> (Qubit, Phase)
qH Xp = (Zp, Pi0 )
qH Xm = (Zm, Pi0 )
qH Yp = (Ym, Pi1q)
qH Ym = (Yp, Pi7q)
qH Zp = (Xp, Pi0 )
qH Zm = (Xm, Pi0 )

-- | Apply a π rotation about X to a qubit.
qX :: Qubit -> (Qubit, Phase)
qX Xp = (Xp, Pi0 )
qX Xm = (Xm, Pi  )
qX Yp = (Ym, Pi1h)
qX Ym = (Yp, Pi3h)
qX Zp = (Zm, Pi0 )
qX Zm = (Zp, Pi0 )

-- | Apply a π rotation about Y to a qubit.
qY :: Qubit -> (Qubit, Phase)
qY Xp = (Xm, Pi3h)
qY Xm = (Xp, Pi1h)
qY Yp = (Yp, Pi0 )
qY Ym = (Ym, Pi  )
qY Zp = (Zm, Pi1h)
qY Zm = (Zp, Pi3h)

-- | Apply a π rotation about Z to a qubit.
qZ :: Qubit -> (Qubit, Phase)
qZ Xp = (Xm, Pi0 )
qZ Xm = (Xp, Pi0 )
qZ Yp = (Ym, Pi0 )
qZ Ym = (Yp, Pi0 )
qZ Zp = (Zp, Pi0 )
qZ Zm = (Zm, Pi  )

-- | Apply a π/2 rotation about Z to a qubit.
qS :: Qubit -> (Qubit, Phase)
qS Xp = (Yp, Pi0 )
qS Xm = (Ym, Pi0 )
qS Yp = (Xm, Pi0 )
qS Ym = (Xp, Pi0 )
qS Zp = (Zp, Pi0 )
qS Zm = (Zm, Pi1h)

-- | A measurement basis.
data Basis = MeasX | MeasY | MeasZ

-- | Return the possible outcomes for a measurement basis, with the plus state
-- first.
basisOutcomes :: Basis -> (Qubit, Qubit)
basisOutcomes MeasX = (Xp, Xm)
basisOutcomes MeasY = (Yp, Ym)
basisOutcomes MeasZ = (Zp, Zm)

-- | Return the plus state associated with a measurement basis.
basisPlus :: Basis -> Qubit
basisPlus = fst . basisOutcomes

-- | Return the minus state associated with a measurement basis.
basisMinus :: Basis -> Qubit
basisMinus = snd . basisOutcomes

-- | Return @True@ if the qubit state is a member of the measurement basis.
basisIsElem :: Basis -> Qubit -> Bool
basisIsElem MeasX Xp = True
basisIsElem MeasX Xm = True
basisIsElem MeasY Yp = True
basisIsElem MeasY Ym = True
basisIsElem MeasZ Zp = True
basisIsElem MeasZ Zm = True
basisIsElem _     _  = False

-- | Randomly select the plus or minus state of a measurement basis with equal
-- probability.
basisGen :: StatefulGen r Maybe => r -> Basis -> Qubit
basisGen rng MeasX = if genb rng then Xp else Xm
basisGen rng MeasY = if genb rng then Yp else Ym
basisGen rng MeasZ = if genb rng then Zp else Zm

