// Asks before an action, in the app's own dialog. Use this instead of window.confirm: the desktop
// window is a webview, so a native dialog is headed by the local address and ignores the theme.

import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

type Confirmation = {
  title: string;
  description?: string;
  /** The confirming button's label: a verb, such as "Delete". */
  action: string;
  danger?: boolean;
};

type Request = Confirmation & { resolve: (confirmed: boolean) => void };

let show: ((request: Request) => void) | undefined;

/** Resolves true when the user confirms, false when they cancel or dismiss the dialog. */
export function confirm(confirmation: Confirmation) {
  return new Promise<boolean>((resolve) => {
    if (show) show({ ...confirmation, resolve });
    else resolve(false);
  });
}

/** Rendered once, at the top of the app. The request stays after closing so the text does not vanish mid-fade. */
export function ConfirmHost() {
  const [request, setRequest] = useState<Request>();
  const [open, setOpen] = useState(false);

  useEffect(() => {
    show = (next) => {
      setRequest((current) => {
        current?.resolve(false);
        return next;
      });
      setOpen(true);
    };
    return () => {
      show = undefined;
    };
  }, []);

  const answer = (confirmed: boolean) => {
    setOpen(false);
    request?.resolve(confirmed);
  };

  return (
    <Dialog open={open} onOpenChange={(next) => !next && answer(false)}>
      <DialogContent showCloseButton={false}>
        <DialogHeader>
          <DialogTitle>{request?.title}</DialogTitle>
          {request?.description && <DialogDescription>{request.description}</DialogDescription>}
        </DialogHeader>
        <DialogFooter>
          <Button variant="outline" onClick={() => answer(false)}>
            Cancel
          </Button>
          <Button variant={request?.danger ? "destructive" : "default"} onClick={() => answer(true)}>
            {request?.action}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
