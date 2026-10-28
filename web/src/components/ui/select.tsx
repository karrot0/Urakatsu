"use client"

import { Select as SelectPrimitive } from "@base-ui/react/select"
import { cn } from "cn"
import { CheckIcon, ChevronDownIcon } from "lucide-react"

import { menuItem, menuPopup } from "@/components/ui/dropdown-menu"

const Select = SelectPrimitive.Root

function SelectGroup({ className, ...props }: SelectPrimitive.Group.Props) {
  return <SelectPrimitive.Group data-slot="select-group" className={cn(className)} {...props} />
}

function SelectValue({ className, ...props }: SelectPrimitive.Value.Props) {
  return <SelectPrimitive.Value data-slot="select-value" className={cn("min-w-0 flex-1 truncate text-left", className)} {...props} />
}

function SelectTrigger({ className, size = "default", children, ...props }: SelectPrimitive.Trigger.Props & { size?: "sm" | "default" }) {
  return (
    <SelectPrimitive.Trigger
      data-slot="select-trigger"
      data-size={size}
      className={cn(
        "flex w-full min-w-0 cursor-default items-center justify-between gap-1 rounded-control bg-surface text-left text-fg shadow-[inset_0_0_0_1px_var(--border)] outline-none transition-[background-color,box-shadow] duration-[20ms] hover:bg-[color-mix(in_oklab,var(--surface),var(--n-950)_3%)] focus-visible:shadow-focus data-popup-open:shadow-focus data-disabled:pointer-events-none data-disabled:opacity-50 data-placeholder:text-fg-3 data-[size=default]:h-8 data-[size=default]:pr-1 data-[size=default]:pl-2 data-[size=default]:text-sm data-[size=sm]:h-[26px] data-[size=sm]:pr-0.5 data-[size=sm]:pl-1 data-[size=sm]:text-xs max-md:data-[size=default]:h-10 max-md:data-[size=default]:pl-3 max-md:data-[size=default]:text-base max-md:data-[size=sm]:h-9 max-md:data-[size=sm]:pl-2.5 max-md:data-[size=sm]:text-base",
        className
      )}
      {...props}
    >
      {children}
      <SelectPrimitive.Icon className="flex shrink-0 text-icon-2">
        <ChevronDownIcon size={14} />
      </SelectPrimitive.Icon>
    </SelectPrimitive.Trigger>
  )
}

function SelectContent({
  className,
  children,
  side = "bottom",
  sideOffset = 4,
  align = "start",
  alignOffset = 0,
  alignItemWithTrigger = false,
  ...props
}: SelectPrimitive.Popup.Props & Pick<SelectPrimitive.Positioner.Props, "align" | "alignOffset" | "side" | "sideOffset" | "alignItemWithTrigger">) {
  return (
    <SelectPrimitive.Portal>
      <SelectPrimitive.Positioner
        side={side}
        sideOffset={sideOffset}
        align={align}
        alignOffset={alignOffset}
        alignItemWithTrigger={alignItemWithTrigger}
        className="isolate z-[60] outline-none"
      >
        <SelectPrimitive.Popup data-slot="select-content" className={cn(menuPopup, "min-w-(--anchor-width)", className)} {...props}>
          <SelectPrimitive.List>{children}</SelectPrimitive.List>
        </SelectPrimitive.Popup>
      </SelectPrimitive.Positioner>
    </SelectPrimitive.Portal>
  )
}

function SelectLabel({ className, ...props }: SelectPrimitive.GroupLabel.Props) {
  return <SelectPrimitive.GroupLabel data-slot="select-label" className={cn("px-2 pt-2 pb-1 text-xs text-fg-3", className)} {...props} />
}

function SelectItem({ className, children, ...props }: SelectPrimitive.Item.Props) {
  return (
    <SelectPrimitive.Item data-slot="select-item" className={cn(menuItem, "pr-8", className)} {...props}>
      <SelectPrimitive.ItemText className="flex min-w-0 flex-1 items-baseline gap-2">{children}</SelectPrimitive.ItemText>
      <SelectPrimitive.ItemIndicator className="pointer-events-none absolute right-2 flex items-center text-brand">
        <CheckIcon size={14} />
      </SelectPrimitive.ItemIndicator>
    </SelectPrimitive.Item>
  )
}

function SelectSeparator({ className, ...props }: SelectPrimitive.Separator.Props) {
  return <SelectPrimitive.Separator data-slot="select-separator" className={cn("-mx-0.5 my-0.5 h-px bg-line-soft", className)} {...props} />
}

export { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectSeparator, SelectTrigger, SelectValue }
