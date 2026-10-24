import { mergeProps } from "@base-ui/react/merge-props"
import { useRender } from "@base-ui/react/use-render"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "cn"

// Tinted background, darker text of the same hue, and a hairline between them.
const badgeVariants = cva(
  "group/badge inline-flex h-5 w-fit shrink-0 items-center gap-1 overflow-hidden rounded-pill px-1.5 text-xs font-medium whitespace-nowrap [&>svg]:pointer-events-none",
  {
    variants: {
      variant: {
        default: "bg-surface-sunk text-fg-2 shadow-[inset_0_0_0_1px_var(--border)]",
        secondary: "bg-surface-fill text-fg-2",
        brand: "bg-brand-weak text-brand shadow-[inset_0_0_0_1px_var(--brand-line)]",
        success: "bg-success-weak text-success shadow-[inset_0_0_0_1px_var(--success-line)]",
        warning: "bg-warning-weak text-warning shadow-[inset_0_0_0_1px_var(--warning-line)]",
        destructive: "bg-danger-weak text-danger shadow-[inset_0_0_0_1px_var(--danger-line)]",
        info: "bg-info-weak text-info shadow-[inset_0_0_0_1px_var(--info-line)]",
        outline: "text-fg shadow-[inset_0_0_0_1px_var(--border)]",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

function Badge({
  className,
  variant = "default",
  render,
  ...props
}: useRender.ComponentProps<"span"> & VariantProps<typeof badgeVariants>) {
  return useRender({
    defaultTagName: "span",
    props: mergeProps<"span">(
      {
        className: cn(badgeVariants({ variant }), className),
      },
      props
    ),
    render,
    state: {
      slot: "badge",
      variant,
    },
  })
}

export { Badge, badgeVariants }
