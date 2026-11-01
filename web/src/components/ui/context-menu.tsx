import { ContextMenu as ContextMenuPrimitive } from "@base-ui/react/context-menu"

import { menuPopup } from "@/components/ui/dropdown-menu"
import { cn } from "cn"

/** Rows and separators are the dropdown menu's: `DropdownMenuItem` and `DropdownMenuSeparator` work inside. */
function ContextMenu({ ...props }: ContextMenuPrimitive.Root.Props) {
  return <ContextMenuPrimitive.Root data-slot="context-menu" {...props} />
}

function ContextMenuTrigger({ ...props }: ContextMenuPrimitive.Trigger.Props) {
  return <ContextMenuPrimitive.Trigger data-slot="context-menu-trigger" {...props} />
}

function ContextMenuContent({ className, ...props }: ContextMenuPrimitive.Popup.Props) {
  return (
    <ContextMenuPrimitive.Portal>
      <ContextMenuPrimitive.Positioner className="isolate z-[60] outline-none">
        <ContextMenuPrimitive.Popup data-slot="context-menu-content" className={cn(menuPopup, className)} {...props} />
      </ContextMenuPrimitive.Positioner>
    </ContextMenuPrimitive.Portal>
  )
}

export { ContextMenu, ContextMenuContent, ContextMenuTrigger }
