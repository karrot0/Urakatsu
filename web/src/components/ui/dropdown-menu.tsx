"use client"

import * as React from "react"
import { Menu as MenuPrimitive } from "@base-ui/react/menu"
import { cn } from "cn"
import { CheckIcon, ChevronRightIcon } from "lucide-react"

import { floatingSurface } from "@/components/ui/popover"

/** Rows shared by menus and selects: 30px tall, control radius, background tint on highlight. */
const menuItem =
  "relative flex min-h-[30px] w-full cursor-default items-center gap-2 rounded-control px-2 py-0.5 text-left text-sm text-fg outline-none select-none transition-colors duration-[20ms] data-highlighted:bg-hover data-disabled:pointer-events-none data-disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 max-md:min-h-touch max-md:rounded-[12px] max-md:px-3 max-md:text-body-lg"

const menuPopup = cn(
  floatingSurface,
  "max-h-[min(var(--available-height,420px),420px)] min-w-28 overflow-y-auto p-0.5"
)

function DropdownMenu({ ...props }: MenuPrimitive.Root.Props) {
  return <MenuPrimitive.Root data-slot="dropdown-menu" {...props} />
}

function DropdownMenuPortal({ ...props }: MenuPrimitive.Portal.Props) {
  return <MenuPrimitive.Portal data-slot="dropdown-menu-portal" {...props} />
}

function DropdownMenuTrigger({ ...props }: MenuPrimitive.Trigger.Props) {
  return <MenuPrimitive.Trigger data-slot="dropdown-menu-trigger" {...props} />
}

function DropdownMenuContent({
  align = "start",
  alignOffset = 0,
  side = "bottom",
  sideOffset = 4,
  className,
  ...props
}: MenuPrimitive.Popup.Props & Pick<MenuPrimitive.Positioner.Props, "align" | "alignOffset" | "side" | "sideOffset">) {
  return (
    <MenuPrimitive.Portal>
      <MenuPrimitive.Positioner className="isolate z-[60] outline-none" align={align} alignOffset={alignOffset} side={side} sideOffset={sideOffset}>
        <MenuPrimitive.Popup data-slot="dropdown-menu-content" className={cn(menuPopup, className)} {...props} />
      </MenuPrimitive.Positioner>
    </MenuPrimitive.Portal>
  )
}

function DropdownMenuGroup({ ...props }: MenuPrimitive.Group.Props) {
  return <MenuPrimitive.Group data-slot="dropdown-menu-group" {...props} />
}

function DropdownMenuLabel({ className, inset, ...props }: MenuPrimitive.GroupLabel.Props & { inset?: boolean }) {
  return (
    <MenuPrimitive.GroupLabel
      data-slot="dropdown-menu-label"
      data-inset={inset}
      className={cn("px-2 pt-2 pb-1 text-xs text-fg-3 data-inset:pl-7", className)}
      {...props}
    />
  )
}

function DropdownMenuItem({
  className,
  inset,
  variant = "default",
  ...props
}: MenuPrimitive.Item.Props & { inset?: boolean; variant?: "default" | "destructive" }) {
  return (
    <MenuPrimitive.Item
      data-slot="dropdown-menu-item"
      data-inset={inset}
      data-variant={variant}
      className={cn(menuItem, "data-inset:pl-7 data-[variant=destructive]:text-danger", className)}
      {...props}
    />
  )
}

function DropdownMenuSub({ ...props }: MenuPrimitive.SubmenuRoot.Props) {
  return <MenuPrimitive.SubmenuRoot data-slot="dropdown-menu-sub" {...props} />
}

function DropdownMenuSubTrigger({ className, inset, children, ...props }: MenuPrimitive.SubmenuTrigger.Props & { inset?: boolean }) {
  return (
    <MenuPrimitive.SubmenuTrigger
      data-slot="dropdown-menu-sub-trigger"
      data-inset={inset}
      className={cn(menuItem, "data-inset:pl-7 data-popup-open:bg-hover", className)}
      {...props}
    >
      {children}
      <ChevronRightIcon size={14} className="ml-auto text-icon-2" />
    </MenuPrimitive.SubmenuTrigger>
  )
}

function DropdownMenuSubContent({
  align = "start",
  alignOffset = -3,
  side = "right",
  sideOffset = 0,
  className,
  ...props
}: React.ComponentProps<typeof DropdownMenuContent>) {
  return (
    <DropdownMenuContent
      data-slot="dropdown-menu-sub-content"
      className={cn("w-auto", className)}
      align={align}
      alignOffset={alignOffset}
      side={side}
      sideOffset={sideOffset}
      {...props}
    />
  )
}

function DropdownMenuCheckboxItem({ className, children, checked, inset, ...props }: MenuPrimitive.CheckboxItem.Props & { inset?: boolean }) {
  return (
    <MenuPrimitive.CheckboxItem
      data-slot="dropdown-menu-checkbox-item"
      data-inset={inset}
      className={cn(menuItem, "pr-8 data-inset:pl-7", className)}
      checked={checked}
      {...props}
    >
      {children}
      <span className="pointer-events-none absolute right-2 flex items-center justify-center text-brand">
        <MenuPrimitive.CheckboxItemIndicator>
          <CheckIcon size={14} />
        </MenuPrimitive.CheckboxItemIndicator>
      </span>
    </MenuPrimitive.CheckboxItem>
  )
}

function DropdownMenuRadioGroup({ ...props }: MenuPrimitive.RadioGroup.Props) {
  return <MenuPrimitive.RadioGroup data-slot="dropdown-menu-radio-group" {...props} />
}

function DropdownMenuRadioItem({ className, children, inset, ...props }: MenuPrimitive.RadioItem.Props & { inset?: boolean }) {
  return (
    <MenuPrimitive.RadioItem
      data-slot="dropdown-menu-radio-item"
      data-inset={inset}
      className={cn(menuItem, "pr-8 data-inset:pl-7", className)}
      {...props}
    >
      {children}
      <span className="pointer-events-none absolute right-2 flex items-center justify-center text-brand">
        <MenuPrimitive.RadioItemIndicator>
          <CheckIcon size={14} />
        </MenuPrimitive.RadioItemIndicator>
      </span>
    </MenuPrimitive.RadioItem>
  )
}

function DropdownMenuSeparator({ className, ...props }: MenuPrimitive.Separator.Props) {
  return <MenuPrimitive.Separator data-slot="dropdown-menu-separator" className={cn("-mx-0.5 my-0.5 h-px bg-line-soft", className)} {...props} />
}

function DropdownMenuShortcut({ className, ...props }: React.ComponentProps<"span">) {
  return <span data-slot="dropdown-menu-shortcut" className={cn("ml-auto font-mono text-xs text-fg-3", className)} {...props} />
}

export {
  DropdownMenu,
  DropdownMenuPortal,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuLabel,
  DropdownMenuItem,
  DropdownMenuCheckboxItem,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuSub,
  DropdownMenuSubTrigger,
  DropdownMenuSubContent,
  menuItem,
  menuPopup,
}
