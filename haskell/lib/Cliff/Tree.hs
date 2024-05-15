-- | Tree-based representation of variable-basis /N/-qubit register states.
--
-- Here, /N/-qubit "register" states are a series of /N/ positions on the Bloch
-- sphere, limited to the usual six cardinal direction: ∣±x⟩, ∣±y⟩, and ∣±z⟩.
-- General states are represented by a quasi-binary tree of these register
-- states, which takes advantage of the fact that, given a single Clifford
-- generator gate (Hadamard, π\/ phase, or CNOT) acting on a single register
-- (basis) state, the action of tehg ate will be to either apply a simple
-- rotation to a single qubit, or to create a Bell state. In the latter case, a
-- basis can always be chosen such that the Bell state can be written as the sum
-- of only two basis states. This is mirror in how non-destructive measurements
-- act on pure states in the creation of mixed states.
--
-- Thus a given `Pure` state is either a single /N/-qubit register state or a
-- superposition of two register states with a relative phase, and its depth in
-- a larger tree encodes the magnitude of both their amplitudes. Likewise, a
-- `State` is either @Pure@ or an even (classical) mixture of two @Pure@s.
--
-- Although states are not principally represented by a naive complex-valued
-- vector or matrix and gate operations are straightforwardly not performed via
-- matrix multiplication, the action of gates and measurements still grow the
-- encoding trees on average, which results in spatial and temporal runtime
-- requirements that are still super-polynomial, making this an /invalid
-- approach for efficient simulation/.
module Cliff.Tree
  ( Qubit (..)
  , qFlip
  , qH
  , qX
  , qY
  , qZ
  , qS
  , Basis (..)
  , basisOutcomes
  , Register
  , regNew
  , regLength
  , regGet
  , regSet
  , regSwap
  , regOp
  , regOpPh
  , Pure
  , pureNew
  , pureIsNull
  , pureIsSingle
  , pureIsSuperpos
  , pureNumTerms
  , pureTerms
  , pureApplyGate
  , pureApplyCircuit
  , pureMeasure
  , State
  , stateNew
  , stateIsEmpty
  , stateIsPure
  , stateIsMixed
  , stateNumTerms
  , stateTerms
  , stateApplyGate
  , stateApplyCircuit
  ) where

import Cliff.Tree.Qubit
import Cliff.Tree.Register
import Cliff.Tree.Pure
import Cliff.Tree.State

