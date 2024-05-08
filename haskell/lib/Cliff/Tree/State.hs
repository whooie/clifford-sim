-- | Definitions for classical mixtures of pure states.
module Cliff.Tree.State
  ( State (..)
  , stateNew
  , stateIsEmpty
  , stateIsPure
  , stateIsMixed
  , stateNumTerms
  , stateTerms
  , stateApplyGate
  , stateApplyCircuit
  , stateMeasure
  ) where

import Numeric.Natural
import Cliff.Gate
import Cliff.Tree.Qubit
import Cliff.Tree.Pure

-- | A classical mixture of /N/-qubit register states.
data State =
  -- | A physically impossible null state.
    Empty
  -- | A single `Pure` state.
  | Pure Pure
  -- | A mixed state comprising the (even) classical mix of two states.
  | Mixed State State

-- | Create a new state consisting of only a single `Pure` initialized to all
-- `Zp`.
stateNew :: Natural -> State
stateNew = Pure . pureNew

-- | Return @True@ if the state is `Empty`.
stateIsEmpty :: State -> Bool
stateIsEmpty Empty = True
stateIsEmpty _     = False

-- | Return @True@ if the state is `Pure`.
stateIsPure :: State -> Bool
stateIsPure (Pure _) = True
stateIsPure _        = False

-- | Return @True@ if the state is `Mixed`.
stateIsMixed :: State -> Bool
stateIsMixed (Mixed _ _) = True
stateIsMixed _           = False

-- | Return the number of pure states in the classical distribution.
--
-- Note that this sum may count some pure states twice.
stateNumTerms :: State -> Int
stateNumTerms Empty = 0
stateNumTerms (Pure _) = 1
stateNumTerms (Mixed l r) = (stateNumTerms l) + (stateNumTerms r)

-- | Return a list of all pure states with associated probabilities.
stateTerms :: State -> [(Pure, Float)]
stateTerms = stateTermsInner 1.0
  where stateTermsInner :: Float -> State -> [(Pure, Float)]
        stateTermsInner depth state =
          case state of
            Empty -> []
            Pure pure -> [(pure, prob)]
              where prob = 0.5 ** depth
            Mixed l r -> termsl ++ termsr
              where termsl = stateTermsInner (depth + 1.0) l
                    termsr = stateTermsInner (depth + 1.0) r

-- | Perform the action of a gate.
stateApplyGate :: Gate -> State -> State
stateApplyGate _ Empty = Empty
stateApplyGate g (Pure pure) = Pure $ pureApplyGate g pure
stateApplyGate g (Mixed l r) = Mixed (stateApplyGate g l) (stateApplyGate g r)

-- | Perform a sequence of gates.
stateApplyCircuit :: [Gate] -> State -> State
stateApplyCircuit gates state =
  foldl (\acc g -> stateApplyGate g acc) state gates

-- | Perform a projective measurement on a single qubit in a given basis.
stateMeasure :: Int -> Basis -> State -> State
stateMeasure _ _ Empty = Empty
stateMeasure k basis (Pure pure) =
  let (op, om) = basisOutcomes basis
      purep = pureMeasure k op pure
      purem = pureMeasure k om pure
   in case (pureIsNull purep, pureIsNull purem) of
        (True,  True ) -> Empty
        (True,  False) -> Pure purem
        (False, True ) -> Pure purep
        (False, False) -> Mixed (Pure purep) (Pure purem)
stateMeasure k basis (Mixed l r) =
  let l' = stateMeasure k basis l
      r' = stateMeasure k basis r
   in case (stateIsEmpty l', stateIsEmpty r') of
        (True,  True ) -> Empty
        (True,  False) -> r'
        (False, True ) -> l'
        (False, False) -> Mixed l' r'

