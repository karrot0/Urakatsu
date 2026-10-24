import { Toggle as TogglePrimitive } from "@base-ui/react/toggle"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "cn"

// "segmented" is the Urakatsu segmented control's button: quiet until pressed, then a raised chip.
const toggleVariants = cva(
  "group/toggle inline-flex cursor-pointer items-center justify-center gap-1 whitespace-nowrap outline-none transition-[background-color,color] duration-[20ms] focus-visible:shadow-focus disabled:pointer-events-none disabled:opacity-45 [&_svg]:pointer-events-none [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default:
          "rounded-control bg-transparent text-fg-2 hover:bg-hover hover:text-fg data-pressed:bg-selected data-pressed:text-fg data-pressed:[&_svg]:text-brand",
        segmented:
          "rounded-[4px] bg-transparent text-fg-2 hover:text-fg data-pressed:bg-surface data-pressed:text-fg data-pressed:shadow-raised",
        outline:
          "rounded-control bg-surface text-fg shadow-[inset_0_0_0_1px_var(--border)] hover:bg-hover data-pressed:bg-selected",
      },
      size: {
        default: "h-6 px-2 text-sm",
        sm: "h-[22px] px-1.5 text-xs",
        lg: "h-8 px-3 text-sm",
        icon: "size-6",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function Toggle({
  className,
  variant = "default",
  size = "default",
  ...props
}: TogglePrimitive.Props & VariantProps<typeof toggleVariants>) {
  return <TogglePrimitive data-slot="toggle" className={cn(toggleVariants({ variant, size, className }))} {...props} />
}

export { Toggle, toggleVariants }
