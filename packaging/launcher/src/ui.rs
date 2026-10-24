//! The installer window: a frameless window showing `ui/installer.html` in the system web view
//! (WebView2 on Windows, WKWebView on macOS, WebKitGTK on Linux).
//!
//! The page sends commands with `window.ipc.postMessage`; the host answers by calling the
//! page's `render(state)`. Work runs on other threads and reaches the window through `Ui`.

use serde_json::Value;
use tao::dpi::LogicalSize;
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy};
use tao::window::{Theme, Window, WindowBuilder};
use wry::{WebContext, WebView, WebViewBuilder};

use crate::platform;

const PAGE: &str = include_str!("../ui/installer.html");

pub enum Message {
    Render(Value),
    Exit,
    Ipc(String),
}

/// A handle work threads use to update the window.
#[derive(Clone)]
pub struct Ui {
    proxy: EventLoopProxy<Message>,
}

impl Ui {
    pub fn render(&self, state: Value) {
        let _ = self.proxy.send_event(Message::Render(state));
    }

    pub fn exit(&self) {
        let _ = self.proxy.send_event(Message::Exit);
    }
}

/// A command from the page: `{"cmd": "...", ...}`.
pub struct Command {
    pub name: String,
    pub data: Value,
}

/// Opens the window on `initial` and runs until `Ui::exit` or the window closes.
/// `on_command` gets every command the window itself does not handle (drag, minimize, close).
pub fn run(initial: Value, mut on_command: impl FnMut(Command, &Ui, &Window) + 'static) -> ! {
    let event_loop = EventLoopBuilder::<Message>::with_user_event().build();
    let ui = Ui {
        proxy: event_loop.create_proxy(),
    };
    let builder = WindowBuilder::new()
        .with_title("Urakatsu Setup")
        .with_inner_size(LogicalSize::new(680.0, 560.0))
        .with_resizable(false)
        .with_maximizable(false)
        .with_decorations(false)
        .with_visible(false);
    let window = with_system_look(builder)
        .build(&event_loop)
        .expect("create the setup window");
    round_corners(&window);

    let mut context = WebContext::new(Some(std::env::temp_dir().join("urakatsu-setup-webview")));
    let page = PAGE.replace("/*STATE*/ null", &initial.to_string());
    let proxy = ui.proxy.clone();
    let dark = platform::dark_mode().unwrap_or_else(|| window.theme() == Theme::Dark);
    let background = if dark {
        (38, 37, 35, 255)
    } else {
        (255, 255, 255, 255)
    };
    let builder = WebViewBuilder::new_with_web_context(&mut context)
        .with_html(page)
        .with_background_color(background)
        .with_devtools(cfg!(debug_assertions))
        .with_ipc_handler(move |request| {
            let _ = proxy.send_event(Message::Ipc(request.body().clone()));
        });
    let webview = build_webview(builder, &window).expect(WEB_VIEW_NEEDED);
    window.set_visible(true);
    window.set_focus();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        let _ = &context; // keeps the web view's profile alive for the window's lifetime
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                on_command(
                    Command {
                        name: "close".into(),
                        data: Value::Null,
                    },
                    &ui,
                    &window,
                );
            }
            Event::UserEvent(Message::Render(state)) => {
                let _ = webview.evaluate_script(&format!("render({state})"));
            }
            Event::UserEvent(Message::Exit) => *control_flow = ControlFlow::Exit,
            Event::UserEvent(Message::Ipc(body)) => {
                let data: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
                let name = data["cmd"].as_str().unwrap_or_default().to_owned();
                match name.as_str() {
                    "drag" => {
                        let _ = window.drag_window();
                    }
                    "minimize" => window.set_minimized(true),
                    _ => on_command(Command { name, data }, &ui, &window),
                }
            }
            _ => {}
        }
    })
}

#[cfg(windows)]
const WEB_VIEW_NEEDED: &str = "create the WebView2 view (Microsoft Edge WebView2 Runtime)";
#[cfg(target_os = "macos")]
const WEB_VIEW_NEEDED: &str = "create the WKWebView view";
#[cfg(all(unix, not(target_os = "macos")))]
const WEB_VIEW_NEEDED: &str = "create the WebKitGTK view (install WebKitGTK 4.1: libwebkit2gtk-4.1)";

/// Windows: the exe's icon on the window and taskbar button, and the shadow a frameless window
/// otherwise loses. The macOS bundle and the Linux menu entry carry the icon there.
#[cfg(windows)]
fn with_system_look(builder: WindowBuilder) -> WindowBuilder {
    use tao::platform::windows::{IconExtWindows, WindowBuilderExtWindows};
    use tao::window::Icon;
    let icon = Icon::from_resource(1, None).ok();
    builder
        .with_undecorated_shadow(true)
        .with_window_icon(icon.clone())
        .with_taskbar_icon(icon)
}

#[cfg(not(windows))]
fn with_system_look(builder: WindowBuilder) -> WindowBuilder {
    builder
}

/// Windows 11 rounds the corners of a frameless window only when asked.
#[cfg(windows)]
fn round_corners(window: &Window) {
    use tao::platform::windows::WindowExtWindows;
    use windows::Win32::Foundation::HWND;
    use windows::Win32::Graphics::Dwm::{
        DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND, DwmSetWindowAttribute,
    };
    unsafe {
        let preference = DWMWCP_ROUND;
        let _ = DwmSetWindowAttribute(
            HWND(window.hwnd() as _),
            DWMWA_WINDOW_CORNER_PREFERENCE,
            &preference as *const _ as _,
            size_of_val(&preference) as u32,
        );
    }
}

#[cfg(not(windows))]
fn round_corners(_window: &Window) {}

/// On Linux the web view goes into the window's GTK box, which works on X11 and Wayland alike.
#[cfg(all(unix, not(target_os = "macos")))]
fn build_webview(builder: WebViewBuilder, window: &Window) -> wry::Result<WebView> {
    use tao::platform::unix::WindowExtUnix;
    use wry::WebViewBuilderExtUnix;
    builder.build_gtk(window.default_vbox().expect("the window's GTK box"))
}

#[cfg(not(all(unix, not(target_os = "macos"))))]
fn build_webview(builder: WebViewBuilder, window: &Window) -> wry::Result<WebView> {
    builder.build(window)
}
