-- | Definitions for registers of qubits.
module Cliff.Tree.Register
  ( Register (..)
  , regNew
  , regLength
  , regGet
  , regSet
  , regSwap
  , regOp
  , regOpPh
  ) where

import Numeric.Natural
import Cliff.Gate
import Cliff.Tree.Qubit

-- | A list of `Qubit`s.
data Register = Register [Qubit] deriving (Eq, Show)

-- | Create a new register of qubits initialized to all `Zp`.
regNew :: Natural -> Register
regNew n = Register [Zp | _ <- [1..n]]

-- | Return the length of a register of qubits.
regLength :: Register -> Int
regLength (Register q) = length q

-- | Index a register of qubits.
regGet :: Int -> Register -> Maybe Qubit
regGet _ (Register []) = Nothing
regGet n (Register (q : t))
  | n == 0    = Just q
  | otherwise = regGet (n - 1) (Register t)

-- | Replace the qubit at an index.
regSet :: Int -> Qubit -> Register -> Register
regSet _ _  (Register []) = Register []
regSet 0 q' (Register (_ : t)) = Register (q' : t)
regSet n q' (Register (q : t)) = Register (q : rec)
  where Register rec = regSet (n - 1) q' (Register t)

-- | Swap the qubits at two indices.
regSwap :: Int -> Int -> Register -> Register
regSwap a b reg = regSwapLeft (min a b) (max a b) reg
  where regSwapLeft :: Int -> Int -> Register -> Register
        regSwapLeft _ _ (Register []) = Register []
        regSwapLeft 0 b (Register (q : t)) =
          let (Register t', mqb) = regSwapRight q (b - 1) (Register t)
           in case mqb of
                Just qb -> Register (qb : t')
                Nothing -> Register (q : t')
        regSwapLeft a b (Register (q : t)) = Register (q : t')
          where Register t' = regSwapLeft (a - 1) (b - 1) (Register t)
        regSwapRight :: Qubit -> Int -> Register -> (Register, Maybe Qubit)
        regSwapRight _  _ (Register []) = (Register [], Nothing)
        regSwapRight qa 0 (Register (q : t)) = (Register (qa : t), Just q)
        regSwapRight qa b (Register (q : t)) = (Register (q : t'), mqb)
          where (Register t', mqb) = regSwapRight qa (b - 1) (Register t)

-- | Perform an arbitrary operation on a given qubit index.
regOp :: Int -> (Qubit -> Qubit) -> Register -> Register
regOp _ _ (Register []) = Register []
regOp 0 f (Register (q : t)) = Register (f q : t)
regOp n f (Register (q : t)) = Register (q : rec)
  where Register rec = regOp (n - 1) f (Register t)

-- | Perform an arbitrary operation incurring a phase on a given qubit index.
regOpPh :: Int -> (Qubit -> (Qubit, Phase)) -> Register -> (Register, Phase)
regOpPh _ _ (Register []) = (Register [], Pi0)
regOpPh 0 f (Register (q : t)) = (Register (q' : t), ph)
  where (q', ph) = f q
regOpPh n f (Register (q : t)) = (Register (q : rec), ph)
  where (Register rec, ph) = regOpPh (n - 1) f (Register t)

