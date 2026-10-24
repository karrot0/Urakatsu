import { Switch as SwitchPrimitive } from "@base-ui/react/switch"
import { cn } from "cn"

function Switch({ className, ...props }: SwitchPrimitive.Root.Props) {
  return (
    <SwitchPrimitive.Root
      data-slot="switch"
      className={cn(
        "group/switch relative inline-flex h-[18px] w-8 shrink-0 cursor-pointer items-center rounded-full bg-line-strong p-0.5 outline-none transition-colors duration-100 focus-visible:shadow-focus data-checked:bg-brand data-disabled:cursor-not-allowed data-disabled:opacity-50",
        className
      )}
      {...props}
    >
      <SwitchPrimitive.Thumb
        data-slot="switch-thumb"
        className="pointer-events-none block size-3.5 rounded-full bg-white shadow-[0_1px_2px_rgba(25,25,25,0.2)] transition-transform duration-150 ease-settle data-checked:translate-x-3.5"
      />
    </SwitchPrimitive.Root>
  )
}

export { Switch }
