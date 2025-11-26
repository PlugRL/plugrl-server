# 🚀 VLARL-Launcher

## 🛠️ Installation

The easiest way to get started is by cloning the repository and using Poetry to handle the dependencies.

1.  **Clone the repository:**

    ```bash
    git clone git@github.com:CTP314/vlarl-launcher.git
    cd vlarl-launcher
    ```

2.  **Install dependencies with Pip:**

    ```bash
    pip install -e .
    ```

### 🚀 Usage

`vlarl-launcher` comes with a command-line utility to launch environment workers.

#### Running a Server

To start a central RL server, use the `vlarl-run-server` command.

```bash
vlarl-run-server
```

examples:

```
vlarl-run-server dummy discrete dummy default
```

This command will launch an server.