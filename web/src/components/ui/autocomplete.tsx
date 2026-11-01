"use client"

import { Autocomplete as AutocompletePrimitive } from "@base-ui/react/autocomplete"
import { cn } from "cn"
import { ChevronDownIcon } from "lucide-react"

import { menuItem, menuPopup } from "@/components/ui/dropdown-menu"
import { fieldChrome } from "@/components/ui/input"

type AutocompleteOption = { value: string; label: string; detail?: string }

/** A text field with a list to pick from: what is typed is the value, on the list or not. */
function Autocomplete({
  label,
  value,
  options,
  placeholder,
  onChange,
}: {
  label: string
  value: string
  options: AutocompleteOption[]
  placeholder?: string
  onChange: (value: string) => void
}) {
  return (
    <AutocompletePrimitive.Root
      items={options}
      value={value}
      onValueChange={onChange}
      itemToStringValue={(option: AutocompleteOption) => option.value}
      filter={null}
      openOnInputClick
    >
      <AutocompletePrimitive.InputGroup className="relative w-full min-w-0">
        <AutocompletePrimitive.Input aria-label={label} placeholder={placeholder} className={cn(fieldChrome, "h-8 pr-8 pl-2 max-md:h-10")} />
        <AutocompletePrimitive.Trigger aria-label={`Show ${label} options`} className="absolute inset-y-0 right-0 flex w-8 items-center justify-center text-icon-2">
          <ChevronDownIcon size={14} />
        </AutocompletePrimitive.Trigger>
      </AutocompletePrimitive.InputGroup>
      <AutocompletePrimitive.Portal>
        <AutocompletePrimitive.Positioner sideOffset={4} align="start" className="isolate z-[60] outline-none">
          <AutocompletePrimitive.Popup data-slot="autocomplete-content" className={cn(menuPopup, "min-w-(--anchor-width)")}>
            <AutocompletePrimitive.List>
              {(option: AutocompleteOption) => (
                <AutocompletePrimitive.Item key={option.value} value={option} className={cn(menuItem, "flex items-baseline justify-between gap-3")}>
                  <span className="min-w-0 truncate">{option.label}</span>
                  {option.detail && <span className="shrink-0 text-xs text-fg-3">{option.detail}</span>}
                </AutocompletePrimitive.Item>
              )}
            </AutocompletePrimitive.List>
          </AutocompletePrimitive.Popup>
        </AutocompletePrimitive.Positioner>
      </AutocompletePrimitive.Portal>
    </AutocompletePrimitive.Root>
  )
}

export { Autocomplete, type AutocompleteOption }
