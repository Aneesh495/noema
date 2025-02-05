import NoemaBridge.Basic

namespace NoemaBridge.Probe

theorem nat_refl (x : Nat) : x = x := by
  rfl

theorem nat_add_zero (x : Nat) : x + 0 = x := by
  rfl

theorem nat_zero_add (x : Nat) : 0 + x = x := by
  rw [Nat.zero_add]

theorem nat_succ_eq (n m : Nat) (h : n = m) : Nat.succ n = Nat.succ m := by
  apply congrArg
  exact h

end NoemaBridge.Probe
