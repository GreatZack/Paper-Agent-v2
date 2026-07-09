"use client";

import { Tooltip } from "radix-ui";
import { forwardRef, type ComponentPropsWithoutRef, type FC } from "react";

type TooltipIconButtonProps = ComponentPropsWithoutRef<"button"> & {
  tooltip: string;
  side?: "top" | "bottom" | "left" | "right";
};

export const TooltipIconButton: FC<TooltipIconButtonProps> = forwardRef<
  HTMLButtonElement,
  TooltipIconButtonProps
>(function TooltipIconButton(
  { tooltip, side = "top", children, ...props },
  ref,
) {
  return (
    <Tooltip.Provider>
      <Tooltip.Root delayDuration={300}>
        <Tooltip.Trigger asChild>
          <button ref={ref} {...props}>
            {children}
          </button>
        </Tooltip.Trigger>
        <Tooltip.Portal>
          <Tooltip.Content
            side={side}
            sideOffset={6}
            className="rounded-lg bg-[#0d0d0d] px-2.5 py-1.5 text-sm text-white dark:bg-[#ececec] dark:text-[#0d0d0d]"
          >
            {tooltip}
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
    </Tooltip.Provider>
  );
});
