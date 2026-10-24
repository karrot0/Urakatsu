import * as React from "react"
import { cn } from "cn"

import { fieldChrome } from "./input"

function Textarea({ className, ...props }: React.ComponentProps<"textarea">) {
  return (
    <textarea
      data-slot="textarea"
      className={cn(fieldChrome, "min-h-[72px] resize-y p-2 leading-5", className)}
      {...props}
    />
  )
}

export { Textarea }
