import { Button as ButtonPrimitive } from "@base-ui/react/button"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "cn"

// Urakatsu look: hairline drawn as an inset shadow (never a border), hover changes only the
// background, focus is the shared two-ring halo. Icons keep the size their lucide `size` gives.
const buttonVariants = cva(
  "group/button inline-flex shrink-0 cursor-pointer items-center justify-center gap-1 rounded-control font-medium whitespace-nowrap no-underline outline-none select-none transition-[background-color,box-shadow,color] duration-[20ms] ease-in disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "bg-brand text-on-brand hover:bg-brand-hover active:bg-brand-press focus-visible:shadow-focus",
        outline:
          "bg-surface text-fg shadow-[inset_0_0_0_1px_var(--border)] hover:bg-hover active:bg-press focus-visible:[box-shadow:inset_0_0_0_1px_var(--border),var(--focus-ring)] [&_svg]:text-icon",
        secondary: "bg-surface-fill text-fg hover:bg-hover active:bg-press focus-visible:shadow-focus [&_svg]:text-icon",
        ghost:
          "bg-transparent text-fg-2 hover:bg-hover hover:text-fg active:bg-press focus-visible:shadow-focus aria-pressed:bg-selected aria-pressed:text-fg data-pressed:bg-selected data-pressed:text-fg [&_svg]:text-icon-2 hover:[&_svg]:text-icon aria-pressed:[&_svg]:text-brand data-pressed:[&_svg]:text-brand",
        destructive: "bg-danger text-on-brand hover:bg-[color-mix(in_oklab,var(--danger)_88%,black)] focus-visible:shadow-focus",
        link: "h-auto! px-0! text-brand underline-offset-4 hover:underline focus-visible:shadow-focus",
      },
      size: {
        default: "h-8 px-3 text-sm",
        sm: "h-7 px-2 text-sm",
        xs: "h-[22px] px-1.5 text-xs",
        lg: "h-9 px-4 text-base",
        icon: "size-8",
        "icon-sm": "size-7",
        "icon-xs": "size-[22px] rounded-pill",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function Button({
  className,
  variant = "default",
  size = "default",
  ...props
}: ButtonPrimitive.Props & VariantProps<typeof buttonVariants>) {
  return (
    <ButtonPrimitive
      data-slot="button"
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Button, buttonVariants }
