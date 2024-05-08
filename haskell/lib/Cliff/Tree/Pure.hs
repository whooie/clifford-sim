-- | Definitions for pure superpositions of register states.
module Cliff.Tree.Pure
  ( Pure (..)
  , pureNew
  , pureIsNull
  , pureIsSingle
  , pureIsSuperpos
  , pureNumTerms
  , pureTerms
  , pureApplyGate
  , pureApplyCircuit
  , pureMeasure
  , Outcome (..)
  ) where

import Data.Complex
import Numeric.Natural
-- import System.Random.Stateful
import Cliff.Gate
-- import Cliff.Random
import Cliff.Tree.Qubit
import Cliff.Tree.Register

-- | A pure state of an /N/-qubit register.
--
-- Constraint to only Clifford gates implies that for any such one- or two-qubit
-- rotation, an appropriate basis can be chosen such that the output state can
-- be written as a superposition of at most two terms. Hence, a pure state is
-- defined recursively here as a quasi-binary tree.
data Pure =
  -- | A physically impossible null state
    Null
  -- | A pure state comprising only a single register state.
  | Single Register
  -- | A pure state comprising the (even) superposition of two pure states with
  -- a relative phase defined as that of the second with respect to the first.
  | Superpos Pure Pure Phase

-- | Create a new qubit register state initialized to all `Zp`.
pureNew :: Natural -> Pure
pureNew = Single . regNew

-- | Return @True@ if the state is `Null`.
pureIsNull :: Pure -> Bool
pureIsNull Null = True
pureIsNull _    = False

-- | Return @True@ if the state is `Single`.
pureIsSingle :: Pure -> Bool
pureIsSingle (Single _) = True
pureIsSingle _          = False

-- | Return @True@ if the state is `Superpos`.
pureIsSuperpos :: Pure -> Bool
pureIsSuperpos (Superpos _ _ _) = True
pureIsSuperpos _                = False

-- | Return the number of terms in the superposition.
--
-- Note that this sum may count some register states twice.
pureNumTerms :: Pure -> Int
pureNumTerms Null = 0
pureNumTerms (Single _) = 1
pureNumTerms (Superpos l r _) = (pureNumTerms l) + (pureNumTerms r)

-- | Return a list of all register states with associated amplitudes.
pureTerms :: Pure -> [(Register, Complex Float)]
pureTerms = pureTermsInner 1.0 Pi0
  where pureTermsInner :: Float -> Phase -> Pure -> [(Register, Complex Float)]
        pureTermsInner depth phase state =
          case state of
            Null -> []
            Single reg -> [(reg, amp)]
              where amp = mkPolar (0.5 ** (depth / 2.0)) (phToFloat phase)
            Superpos l r ph -> termsl ++ termsr
              where termsl = pureTermsInner (depth + 1.0) phase l
                    termsr = pureTermsInner (depth + 1.0) (phase @+ ph) r

-- | Perform the action of a gate.
pureApplyGate :: Gate -> Pure -> Pure
pureApplyGate gate state = fst $ pureApplyGateInner gate state
  where pureApplyGateInner :: Gate -> Pure -> (Pure, Maybe Phase)
        pureApplyGateInner _ Null = (Null, Nothing)
        pureApplyGateInner g (Single reg) =
          case g of
            H k -> (Single reg', Just ph)
              where (reg', ph) = regOpPh k qH reg
            X k -> (Single reg', Just ph)
              where (reg', ph) = regOpPh k qX reg
            Y k -> (Single reg', Just ph)
              where (reg', ph) = regOpPh k qY reg
            Z k -> (Single reg', Just ph)
              where (reg', ph) = regOpPh k qZ reg
            S k -> (Single reg', Just ph)
              where (reg', ph) = regOpPh k qS reg
            CX a b -> if a == b then (Single reg, Just Pi0) else
              case (regGet a reg, regGet b reg) of
                (Nothing, Nothing) -> (Single reg, Just Pi0)
                (Nothing, _) -> (Single reg, Just Pi0)
                (_, Nothing) -> (Single reg, Just Pi0)
                --
                (Just Xp, Just Xp) -> (Single reg, Just Pi0)
                (Just Xp, Just Xm) -> (Single $ regOp a qFlip reg, Just Pi0)
                (Just Xm, Just Xp) -> (Single reg, Just Pi0)
                (Just Xm, Just Xm) -> (Single $ regOp a qFlip reg, Just Pi0)
                --
                (Just Xp, Just Yp) -> (Superpos l r Pi1h, Just Pi0)
                  where l = Single $ regSet a Zp reg
                        r = Single $ regSet a Zm $ regOp b qFlip reg
                (Just Xp, Just Ym) -> (Superpos l r Pi3h, Just Pi0)
                  where l = Single $ regSet a Zp reg
                        r = Single $ regSet a Zm $ regOp b qFlip reg
                (Just Xm, Just Yp) -> (Superpos l r Pi3h, Just Pi0)
                  where l = Single $ regSet a Zp reg
                        r = Single $ regSet a Zm $ regOp b qFlip reg
                (Just Xm, Just Ym) -> (Superpos l r Pi1h, Just Pi0)
                  where l = Single $ regSet a Zp reg
                        r = Single $ regSet a Zm $ regOp b qFlip reg
                --
                (Just Xp, Just Zp) -> (Superpos l r Pi0, Just Pi0)
                  where l = Single $ regSet a Zp reg
                        r = Single $ regSet a Zm $ regOp b qFlip reg
                (Just Xp, Just Zm) -> (Superpos l r Pi0, Just Pi0)
                  where l = Single $ regSet a Zp reg
                        r = Single $ regSet a Zm $ regOp b qFlip reg
                (Just Xm, Just Zp) -> (Superpos l r Pi,  Just Pi0)
                  where l = Single $ regSet a Zp reg
                        r = Single $ regSet a Zm $ regOp b qFlip reg
                (Just Xm, Just Zm) -> (Superpos l r Pi,  Just Pi0)
                  where l = Single $ regSet a Zp reg
                        r = Single $ regSet a Zm $ regOp b qFlip reg
                --
                (Just Yp, Just Xp) -> (Single reg, Just Pi0)
                (Just Yp, Just Xm) -> (Single $ regOp a qFlip reg, Just Pi0)
                (Just Ym, Just Xp) -> (Single reg, Just Pi0)
                (Just Ym, Just Xm) -> (Single $ regOp a qFlip reg, Just Pi0)
                --
                (Just Yp, Just Yp) -> (Superpos l r Pi,  Just Pi0)
                  where l = Single $ regSet a Zp reg
                        r = Single $ regSet a Zm $ regOp b qFlip reg
                (Just Yp, Just Ym) -> (Superpos l r Pi0, Just Pi0)
                  where l = Single $ regSet a Zp reg
                        r = Single $ regSet a Zm $ regOp b qFlip reg
                (Just Ym, Just Yp) -> (Superpos l r Pi0, Just Pi0)
                  where l = Single $ regSet a Zp reg
                        r = Single $ regSet a Zm $ regOp b qFlip reg
                (Just Ym, Just Ym) -> (Superpos l r Pi,  Just Pi0)
                  where l = Single $ regSet a Zp reg
                        r = Single $ regSet a Zm $ regOp b qFlip reg
                --
                (Just Yp, Just Zp) -> (Superpos l r Pi0, Just Pi1h)
                  where l = Single $ regSet a Zp reg
                        r = Single $ regSet a Zm $ regOp b qFlip reg
                (Just Yp, Just Zm) -> (Superpos l r Pi0, Just Pi1h)
                  where l = Single $ regSet a Zp reg
                        r = Single $ regSet a Zm $ regOp b qFlip reg
                (Just Ym, Just Zp) -> (Superpos l r Pi,  Just Pi3h)
                  where l = Single $ regSet a Zp reg
                        r = Single $ regSet a Zm $ regOp b qFlip reg
                (Just Ym, Just Zm) -> (Superpos l r Pi,  Just Pi3h)
                  where l = Single $ regSet a Zp reg
                        r = Single $ regSet a Zm $ regOp b qFlip reg
                --
                (Just Zp, Just Xp) -> (Single reg, Just Pi0)
                (Just Zp, Just Xm) -> (Single reg, Just Pi0)
                (Just Zm, Just Xp) -> (Single reg, Just Pi0)
                (Just Zm, Just Xm) -> (Single reg, Just Pi )
                --
                (Just Zp, Just Yp) -> (Single reg, Just Pi0)
                (Just Zp, Just Ym) -> (Single reg, Just Pi0)
                (Just Zm, Just Yp) -> (Single $ regOp b qFlip reg, Just Pi1h)
                (Just Zm, Just Ym) -> (Single $ regOp b qFlip reg, Just Pi3h)
                --
                (Just Zp, Just Zp) -> (Single reg, Just Pi0)
                (Just Zp, Just Zm) -> (Single reg, Just Pi0)
                (Just Zm, Just Zp) -> (Single $ regOp b qFlip reg, Just Pi0)
                (Just Zm, Just Zm) -> (Single $ regOp b qFlip reg, Just Pi0)
            Swap a b -> (Single reg', Just Pi0)
              where reg' = regSwap a b reg
        pureApplyGateInner g (Superpos l r ph) =
          case (pureApplyGateInner g l, pureApplyGateInner g r) of
            ((l', Just phl), (r', Just phr)) ->
              (Superpos l' r' (phr @- phl), Just phl)
            ((l', Just phl), (r', Nothing)) ->
              (Superpos l' r' (ph @- phl), Just phl)
            ((l', Nothing), (r', Just phr)) ->
              (Superpos l' r' (ph @+ phr), Just Pi0)
            ((_, Nothing), (_, Nothing)) ->
              (Null, Nothing)

-- | Perform a series of gates
pureApplyCircuit :: [Gate] -> Pure -> Pure
pureApplyCircuit gates state = foldl (\acc g -> pureApplyGate g acc) state gates

-- | Perform a projective measurement on a single qubit, post-selected to a
-- particular outcome.
pureMeasure :: Int -> Qubit -> Pure -> Pure
pureMeasure k outcome state = fst $ pureMeasureInner k outcome state
  where pureMeasureInner :: Int -> Qubit -> Pure -> (Pure, Maybe Phase)
        pureMeasureInner _ _ Null = (Null, Nothing)
        pureMeasureInner k outcome (Single reg) =
          case (regGet k reg, outcome) of
            (Just Xp, Xm) -> (Null, Nothing)
            (Just Xm, Xp) -> (Null, Nothing)
            (Just Yp, Ym) -> (Null, Nothing)
            (Just Ym, Yp) -> (Null, Nothing)
            (Just Zp, Zm) -> (Null, Nothing)
            (Just Zm, Zp) -> (Null, Nothing)
            (Nothing, _)  -> (Null, Nothing)
            _ -> (Single $ regSet k outcome reg, Just Pi0)
        pureMeasureInner k outcome (Superpos l r ph) =
          case (pureMeasureInner k outcome l, pureMeasureInner k outcome r) of
            ((_, Nothing), (_, Nothing)) -> (Null, Nothing)
            ((_, Nothing), (r, Just phr)) -> (r, Just $ ph @+ phr)
            ((l, Just phl), (_, Nothing)) -> (l, Just phl)
            ((l, Just phl), (_, Just phr)) ->
              (Superpos l r (ph @+ phr @- phl), Just phl)

-- -- | Perform a (possibly) randomized projective measurement on a single qubit,
-- -- returning the outcome as well as whether the measurement was deterministic
-- -- or random.
-- pureMeasureRng
--   :: StatefulGen r Maybe
--   => r -> Int -> Basis -> Pure -> (Pure, Maybe Outcome)
-- pureMeasureRng _ _ _ Null = (Null, Nothing)
-- pureMeasureRng rng k basis (Single reg) =
--   let maybeQ = regGet k reg
--    in case fmap (\q -> (basisIsElem basis q, q)) maybeQ of
--         Nothing -> (Null, Nothing)
--         Just (True, q) -> (Single reg, Just $ Det q)
--         Just (False, q) -> (Single reg', Just $ Rand q')
--           where q' = basisGen rng basis
--                 reg' = regSet k q' reg
-- pureMeasureRng rng k basis (Superpos l r ph) =
--   pureMeasureRngInner rng k basis Nothing (Superpos l r ph)
--
-- pureMeasureRngInner
--   :: StatefulGen r Maybe
--
--
--   let (l', maybeOutL) = pureMeasureRng rng k basis l
--       (r', maybeOutR) = pureMeasureRng rng k basis r
--    in case (maybeOutL, maybeOutR) of
--         (Nothing, Nothing) -> (Null, Nothing)
--         (Nothing, Just outr) -> (r', Just outr)
--         (Just outl, Nothing) -> (

-- | The outcome of a single measurement performed on a `Pure`.
data Outcome =
  -- | A deterministic measurement.
    Det Qubit
  -- | A randomized measurement.
  | Rand Qubit

