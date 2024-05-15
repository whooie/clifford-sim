-- | Definitions for registers of qubits.
module Cliff.Tree.Register
  ( Register
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
type Register = [Qubit]

-- | Create a new register of qubits initialized to all `Zp`.
regNew :: Natural -> Register
regNew n = [Zp | _ <- [1..n]]

-- | Return the length of a register of qubits.
regLength :: Register -> Int
regLength q = length q

-- | Index a register of qubits.
regGet :: Int -> Register -> Maybe Qubit
regGet _ [] = Nothing
regGet n (q : t)
  | n == 0    = Just q
  | otherwise = regGet (n - 1) t

-- | Replace the qubit at an index.
regSet :: Int -> Qubit -> Register -> Register
regSet _ _  [] = []
regSet 0 q' (_ : t) = q' : t
regSet n q' (q : t) = q : rec
  where rec = regSet (n - 1) q' t

-- | Swap the qubits at two indices.
regSwap :: Int -> Int -> Register -> Register
regSwap a b reg = regSwapLeft (min a b) (max a b) reg
  where regSwapLeft :: Int -> Int -> Register -> Register
        regSwapLeft _ _ [] = []
        regSwapLeft 0 b (q : t) =
          let (t', mqb) = regSwapRight q (b - 1) t
           in case mqb of
                Just qb -> qb : t'
                Nothing -> q : t'
        regSwapLeft a b (q : t) = q : t'
          where t' = regSwapLeft (a - 1) (b - 1) t
        regSwapRight :: Qubit -> Int -> Register -> (Register, Maybe Qubit)
        regSwapRight _  _ [] = ([], Nothing)
        regSwapRight qa 0 (q : t) = (qa : t, Just q)
        regSwapRight qa b (q : t) = (q : t', mqb)
          where (t', mqb) = regSwapRight qa (b - 1) t

-- | Perform an arbitrary operation on a given qubit index.
regOp :: Int -> (Qubit -> Qubit) -> Register -> Register
regOp _ _ [] = []
regOp 0 f (q : t) = f q : t
regOp n f (q : t) = q : rec
  where rec = regOp (n - 1) f t

-- | Perform an arbitrary operation incurring a phase on a given qubit index.
regOpPh :: Int -> (Qubit -> (Qubit, Phase)) -> Register -> (Register, Phase)
regOpPh _ _ [] = ([], Pi0)
regOpPh 0 f (q : t) = (q' : t, ph)
  where (q', ph) = f q
regOpPh n f (q : t) = (q : rec, ph)
  where (rec, ph) = regOpPh (n - 1) f t

