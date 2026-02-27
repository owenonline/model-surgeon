import * as vscode from 'vscode';

export class Logger {
  private static channel: vscode.OutputChannel;

  public static initialize() {
    if (!this.channel) {
      this.channel = vscode.window.createOutputChannel('Model Surgeon');
    }
  }

  public static log(message: string, operation?: string) {
    const timestamp = new Date().toISOString();
    const prefix = operation ? `[${operation}] ` : '';
    this.channel.appendLine(`${timestamp} - ${prefix}${message}`);
  }

  public static error(message: string, error?: unknown) {
    const timestamp = new Date().toISOString();
    const errStr = error instanceof Error ? error.message : String(error);
    this.channel.appendLine(`${timestamp} - [ERROR] ${message} ${errStr}`);
  }

  public static showOutput() {
    this.channel.show(true);
  }
}
