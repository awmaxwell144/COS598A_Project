# Start with ollama image as the base image
FROM ollama/ollama:latest

# Set environment variables
ENV USER am8792
ENV VSCODE_INSTALL_DIR /opt/VSCode-linux-x64
ENV VSCODE_BINARY_URL "https://code.visualstudio.com/sha/download?build=stable&os=linux-x64"
WORKDIR /root

# Copy requirements and entrypoint
COPY requirements.txt .
COPY req_with_deps.txt .
COPY entrypoint.sh /usr/bin/
RUN chmod +x /usr/bin/entrypoint.sh

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    git                \
    curl               \
    python3            \
    python3-pip        \
    libnss3            \
    libatk1.0-0        \
    libatk-bridge2.0-0 \
    libdrm2            \
    libgtk-3-0         \
    libgbm1            \
    libasound2         \
    libgl1-mesa-dev    \
    libfftw3-dev       \
    quantum-espresso   \
    openbabel          \
    swig               \
    freeglut3-dev      \
    libgles2-mesa-dev  \
    xvfb               \
    zsh                \
    fonts-powerline    \
    emacs              \
    gmsh               \
    ffmpeg             \
    ca-certificates    \
    && apt-get clean   \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --no-deps -r requirements.txt 

RUN pip install -U "jax[cuda12]"

# Install VSCode if needed
RUN mkdir -p $VSCODE_INSTALL_DIR/data
RUN curl -L $VSCODE_BINARY_URL -o /tmp/vscode-linux-x64.tar.gz \
    && tar -zxvf /tmp/vscode-linux-x64.tar.gz --directory $VSCODE_INSTALL_DIR --strip-components=1

# Add user
RUN addgroup --gid 1000 $USER \
  && adduser --uid 1000 --ingroup $USER --home /home/$USER --shell /usr/bin/zsh --disabled-password --gecos "" $USER \
  && chown -R $USER:$USER /home/$USER \
  && chown -R $USER:$USER $VSCODE_INSTALL_DIR

# Install fixuid if needed
RUN USER=$USER \
  && GROUP=$USER \
  && curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.6.0/fixuid-0.6.0-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - \
  && chown root:root /usr/local/bin/fixuid \
  && chmod 4755 /usr/local/bin/fixuid \
  && mkdir -p /etc/fixuid \
  && printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml

USER $USER:$GROUP
ENTRYPOINT ["/usr/bin/entrypoint.sh"]
WORKDIR /home/$USER
