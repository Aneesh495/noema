import NoemaBridge.Basic

namespace NoemaBridge.Algebra

structure Monoid (α : Type) where
  mul : α → α → α
  one : α
  mul_assoc : ∀ a b c : α, mul (mul a b) c = mul a (mul b c)
  one_mul : ∀ a : α, mul one a = a
  mul_one : ∀ a : α, mul a one = a

structure Group (α : Type) extends Monoid α where
  inv : α → α
  inv_mul_cancel : ∀ a : α, mul (inv a) a = one
  mul_inv_cancel : ∀ a : α, mul a (inv a) = one

namespace Group

variable {α : Type} (G : Group α)

theorem left_cancel (a b c : α) (h : G.mul a b = G.mul a c) : b = c := by
  have h2 : G.mul (G.inv a) (G.mul a b) = G.mul (G.inv a) (G.mul a c) := by rw [h]
  rw [← G.mul_assoc, ← G.mul_assoc, G.inv_mul_cancel, G.one_mul, G.one_mul] at h2
  exact h2

theorem right_cancel (a b c : α) (h : G.mul b a = G.mul c a) : b = c := by
  have h2 : G.mul (G.mul b a) (G.inv a) = G.mul (G.mul c a) (G.inv a) := by rw [h]
  rw [G.mul_assoc, G.mul_assoc, G.mul_inv_cancel, G.mul_one, G.mul_one] at h2
  exact h2

theorem unique_identity (e : α) (h : ∀ x : α, G.mul e x = x) : e = G.one := by
  have h1 := h G.one
  rw [G.mul_one] at h1
  exact h1

theorem inv_inv (a : α) : G.inv (G.inv a) = a := by
  apply G.left_cancel (G.inv a)
  rw [G.inv_mul_cancel, G.mul_inv_cancel]

theorem inv_one : G.inv G.one = G.one := by
  apply G.left_cancel G.one
  rw [G.mul_inv_cancel, G.mul_one]

theorem inv_mul_rev (a b : α) : G.inv (G.mul a b) = G.mul (G.inv b) (G.inv a) := by
  apply G.left_cancel (G.mul a b)
  rw [G.mul_inv_cancel]
  rw [G.mul_assoc, ← G.mul_assoc b, G.mul_inv_cancel, G.one_mul, G.mul_inv_cancel]

end Group

structure MonoidHom {α β : Type} (M : Monoid α) (N : Monoid β) where
  toFun : α → β
  map_mul' : ∀ a b : α, toFun (M.mul a b) = N.mul (toFun a) (toFun b)
  map_one' : toFun M.one = N.one

def compHom {α β γ : Type} {M : Monoid α} {N : Monoid β} {P : Monoid γ}
    (g : MonoidHom N P) (f : MonoidHom M N) : MonoidHom M P where
  toFun := fun x => g.toFun (f.toFun x)
  map_mul' := by
    intro a b
    dsimp
    rw [f.map_mul', g.map_mul']
  map_one' := by
    dsimp
    rw [f.map_one', g.map_one']

def idHom {α : Type} (M : Monoid α) : MonoidHom M M where
  toFun := fun x => x
  map_mul' := by intro a b; rfl
  map_one' := by rfl

end NoemaBridge.Algebra
