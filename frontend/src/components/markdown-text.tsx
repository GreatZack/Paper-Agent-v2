"use client";

import "@assistant-ui/react-markdown/styles/dot.css";

import {
  MarkdownTextPrimitive,
} from "@assistant-ui/react-markdown";
import remarkGfm from "remark-gfm";
import { FC, memo } from "react";

const MarkdownTextImpl: FC = () => {
  return (
    <MarkdownTextPrimitive
      remarkPlugins={[remarkGfm]}
      className="aui-md"
    />
  );
};

export const MarkdownText = memo(MarkdownTextImpl);
