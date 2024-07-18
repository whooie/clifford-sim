{- stack script
   --snapshot lts-22.21
-}
module Main (main) where

import Control.Monad (foldM)
import Data.Bits (shiftR, (.&.))
import Data.Function ((&))

data BinRatio = BinRatio { num :: Int, pow :: Int }

makeRatio :: Int -> Int -> BinRatio
makeRatio pw nm = BinRatio { num = nm, pow = pw }

floatFromInt :: Int -> Float
floatFromInt = fromInteger . toInteger

(//) :: Int -> Int -> Int
a // b = floor $ af / bf
  where af = floatFromInt a
        bf = floatFromInt b

toFloat :: Int -> Int -> Float
toFloat num pow = numf / (2.0 ** powf)
  where numf = floatFromInt num
        powf = floatFromInt pow

approximateProb :: Float -> Float -> BinRatio
approximateProb = doApproximateProb 0 1 0 15

doApproximateProb :: Int -> Int -> Int -> Int -> Float -> Float -> BinRatio
doApproximateProb l r pw pwMax p eps =
  let distL = abs $ (toFloat l pw) - p
      distR = abs $ (toFloat r pw) - p
  in
  if distL < eps || distR < eps || pw == pwMax
    then makeRatio pw $ if distL < distR then l else r
  else if distL < distR
    then doApproximateProb (2 * l) (2 * r - 1) (pw + 1) pwMax p eps
  else
    doApproximateProb (2 * l + 1) (2 * r) (pw + 1) pwMax p eps

getBitsPos :: Int -> Int -> Int
getBitsPos total k =
  if total == 2
    then k .&. 1
  else if k .&. 1 == 1
    then (total // 2) + getBitsPos (total // 2) (k `shiftR` 1)
  else
    getBitsPos (total // 2) (k `shiftR` 1)

listReplace :: a -> Int -> [a] -> [a]
listReplace _       _ []      = []
listReplace newItem 0 (h : t) = newItem : t
listReplace newItem k (h : t) = h : listReplace newItem (k - 1) t

makePattern :: BinRatio -> [Bool]
makePattern (BinRatio { num = nm, pow = pw }) =
  let pwf = floatFromInt pw
      n = round $ 2.0 ** pwf
      initList = [ False | _ <- [0 .. n - 1] ]
  in
  [0 .. nm - 1]
  & map (getBitsPos n)
  & foldl (flip (listReplace True)) initList

main :: IO ()
main = do
  let doit = putStrLn . show . makePattern . makeRatio 3
  [0..8] & foldM (\_ k -> doit k) ()

