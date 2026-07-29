export type PipelineStatus =
  | "idle"
  | "connecting"
  | "running"
  | "completed"
  | "failed";

export interface PipelineMessage {
  step: string;
  state: string;
  data: unknown;
}

export interface PipelineCallbacks {
  onProgress: (step: string, state: string, data: unknown) => void;
  onReport: (report: string) => void;
  onError: (error: string) => void;
  onStatusChange: (status: PipelineStatus) => void;
}

export class PipelineSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private callbacks: PipelineCallbacks;
  private intentionalClose = false;
  private failureNotified = false;

  constructor(url: string, callbacks: PipelineCallbacks) {
    this.url = url;
    this.callbacks = callbacks;
  }

  connect(query: string, maxPapers?: number) {
    this.intentionalClose = false;
    this.failureNotified = false;
    this.callbacks.onStatusChange("connecting");
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      this.callbacks.onStatusChange("running");
      const payload: Record<string, unknown> = { action: "start", query };
      if (maxPapers !== undefined) payload.max_papers = maxPapers;
      this.ws!.send(JSON.stringify(payload));
    };

    this.ws.onmessage = (event) => {
      try {
        const msg: PipelineMessage = JSON.parse(event.data);

        if (msg.step === "report") {
          this.callbacks.onStatusChange("completed");
          this.callbacks.onReport(msg.data as string);
          this.close();
        } else if (msg.step === "failed") {
          this.callbacks.onStatusChange("failed");
          this.callbacks.onError(msg.data as string);
          this.close();
        } else {
          this.callbacks.onProgress(msg.step, msg.state, msg.data);
        }
      } catch {
        this.callbacks.onError("Failed to parse server message");
      }
    };

    this.ws.onerror = () => this.notifyFailure("WebSocket connection error");

    this.ws.onclose = () => {
      this.ws = null;
      if (!this.intentionalClose) {
        this.notifyFailure(
          "与服务器的连接已中断，请稍后重试。长任务运行期间请保持页面打开。",
        );
      }
    };
  }

  private notifyFailure(message: string) {
    if (this.failureNotified || this.intentionalClose) return;
    this.failureNotified = true;
    this.callbacks.onStatusChange("failed");
    this.callbacks.onError(message);
  }

  close() {
    if (this.ws) {
      this.intentionalClose = true;
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
  }
}
