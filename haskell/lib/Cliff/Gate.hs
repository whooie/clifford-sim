-- | Gates whose operations belong to the /N/-qubit Clifford group.
--
-- See also: <https://en.wikipedia.org/wiki/Clifford_gates>
module Cliff.Gate
  ( Phase (..)
  , phToInt
  , phFromInt
  , phToFloat
  , (@+)
  , (@-)
  , phNeg
  , Gate (..)
  , gateIsH
  , gateIsX
  , gateIsY
  , gateIsZ
  , gateIsS
  , gateIsCX
  , gateIsSwap
  ) where

-- | The argument of a complex phase factor, limited to integer multiples of
-- π/4.
data Phase = Pi0 | Pi1q | Pi1h | Pi3q | Pi | Pi5q | Pi3h | Pi7q
  deriving (Eq)

instance Enum Phase where
  fromEnum ph =
    case ph of
      Pi0  -> 0
      Pi1q -> 1
      Pi1h -> 2
      Pi3q -> 3
      Pi   -> 4
      Pi5q -> 5
      Pi3h -> 6
      Pi7q -> 7
  toEnum m =
    case m `mod` 8 of
      0 -> Pi0
      1 -> Pi1q
      2 -> Pi1h
      3 -> Pi3q
      4 -> Pi
      5 -> Pi5q
      6 -> Pi3h
      7 -> Pi7q
      _ -> error "unreachable"

instance Show Phase where
  show ph =
    case ph of
      Pi0  -> "+1"
      Pi1q -> "+e^iπ/4"
      Pi1h -> "+i"
      Pi3q -> "+e^i3π/4"
      Pi   -> "-1"
      Pi5q -> "+e^i5π/4"
      Pi3h -> "-i"
      Pi7q -> "+e^i7π/4"

-- | Convert a phase to the bare multiple of π/4.
phToInt :: Phase -> Int
phToInt = fromEnum

-- | Convert a bare multiple of π/4 to a phase.
phFromInt :: Int -> Phase
phFromInt = toEnum

-- | Convert to an exact value.
phToFloat :: Phase -> Float
phToFloat ph =
  case ph of
    Pi0  -> 0.00
    Pi1q -> 0.25 * pi
    Pi1h -> 0.50 * pi
    Pi3q -> 0.75 * pi
    Pi   -> pi
    Pi5q -> 1.25 * pi
    Pi3h -> 1.50 * pi
    Pi7q -> 1.75 * pi

-- | Add two phases modulo π.
(@+) :: Phase -> Phase -> Phase
(@+) ph1 ph2 = phFromInt $ (phToInt ph1) + (phToInt ph2)

-- | Subtract two phases modulo π.
(@-) :: Phase -> Phase -> Phase
(@-) ph1 ph2 = phFromInt $ (phToInt ph1) - (phToInt ph2)

-- | Flip the sign of a phase modulo π.
phNeg :: Phase -> Phase
phNeg ph = phFromInt $ -(phToInt ph)

-- | Description of a single gate.
data Gate =
  -- | Hadamard
    H Int
  -- | π rotation about X
  | X Int
  -- | π rotation about Y
  | Y Int
  -- | π rotation about Z
  | Z Int
  -- | π/2 rotation about Z
  | S Int
  -- | Z-controlled π rotation about X. The first qubit index is the control.
  | CX Int Int
  -- | Swap
  | Swap Int Int
  deriving (Eq, Show)

-- | Return @True@ if the gate is `H`.
gateIsH :: Gate -> Bool
gateIsH (H _) = True
gateIsH _     = False

-- | Return @True@ if the gate is `X`.
gateIsX :: Gate -> Bool
gateIsX (X _) = True
gateIsX _     = False

-- | Return @True@ if the gate is `Y`.
gateIsY :: Gate -> Bool
gateIsY (Y _) = True
gateIsY _     = False

-- | Return @True@ if the gate is `Z`.
gateIsZ :: Gate -> Bool
gateIsZ (Z _) = True
gateIsZ _     = False

-- | Return @True@ if the gate is `S`.
gateIsS :: Gate -> Bool
gateIsS (S _) = True
gateIsS _     = False

-- | Return @True@ if the gate is `CX`.
gateIsCX :: Gate -> Bool
gateIsCX (CX _ _) = True
gateIsCX _        = False

-- | Return @True@ if the gate is `Swap`.
gateIsSwap :: Gate -> Bool
gateIsSwap (Swap _ _) = True
gateIsSwap _          = False

