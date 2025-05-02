import NoemaBridge.Basic

namespace NoemaBridge.Order

structure PartialOrder (α : Type) where
  le : α → α → Prop
  le_refl : ∀ a : α, le a a
  le_trans : ∀ a b c : α, le a b → le b c → le a c
  le_antisymm : ∀ a b : α, le a b → le b a → a = b

structure SemilatticeInf (α : Type) extends PartialOrder α where
  inf : α → α → α
  inf_le_left : ∀ a b : α, le (inf a b) a
  inf_le_right : ∀ a b : α, le (inf a b) b
  le_inf : ∀ a b c : α, le c a → le c b → le c (inf a b)

namespace SemilatticeInf

variable {α : Type} (L : SemilatticeInf α)

theorem inf_comm (a b : α) : L.inf a b = L.inf b a := by
  apply L.le_antisymm
  · apply L.le_inf
    · exact L.inf_le_right a b
    · exact L.inf_le_left a b
  · apply L.le_inf
    · exact L.inf_le_right b a
    · exact L.inf_le_left b a

theorem inf_idemp (a : α) : L.inf a a = a := by
  apply L.le_antisymm
  · exact L.inf_le_left a a
  · apply L.le_inf
    · exact L.le_refl a
    · exact L.le_refl a

end SemilatticeInf

structure SemilatticeSup (α : Type) extends PartialOrder α where
  sup : α → α → α
  le_sup_left : ∀ a b : α, le a (sup a b)
  le_sup_right : ∀ a b : α, le b (sup a b)
  sup_le : ∀ a b c : α, le a c → le b c → le (sup a b) c

namespace SemilatticeSup

variable {α : Type} (L : SemilatticeSup α)

theorem sup_comm (a b : α) : L.sup a b = L.sup b a := by
  apply L.le_antisymm
  · apply L.sup_le
    · exact L.le_sup_right b a
    · exact L.le_sup_left b a
  · apply L.sup_le
    · exact L.le_sup_right a b
    · exact L.le_sup_left a b

theorem sup_idemp (a : α) : L.sup a a = a := by
  apply L.le_antisymm
  · apply L.sup_le
    · exact L.le_refl a
    · exact L.le_refl a
  · exact L.le_sup_left a a

end SemilatticeSup

def Monotone {α β : Type} (P : PartialOrder α) (Q : PartialOrder β) (f : α → β) : Prop :=
  ∀ a b : α, P.le a b → Q.le (f a) (f b)

theorem monotone_comp {α β γ : Type} {P : PartialOrder α} {Q : PartialOrder β} {R : PartialOrder γ}
    {f : α → β} {g : β → γ} (hf : Monotone P Q f) (hg : Monotone Q R g) :
    Monotone P R (fun x => g (f x)) := by
  intro a b hab
  exact hg (f a) (f b) (hf a b hab)

theorem monotone_id {α : Type} (P : PartialOrder α) : Monotone P P (fun x => x) := by
  intro a b hab
  exact hab

end NoemaBridge.Order
