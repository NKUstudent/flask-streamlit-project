document.getElementById('send-button').addEventListener('click', sendMessage);
document.getElementById('user-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function sendMessage() {
    const userInput = document.getElementById('user-input');
    const message = userInput.value.trim();
    if (message) {
        document.getElementById('intro-box').style.display = 'none'; // Hide intro box when a message is sent
        appendMessage('user', message);
        userInput.value = '';
        // Simulate bot response
        setTimeout(() => {
            appendMessage('bot', getBotResponse(message));
        }, 500);
    }
}

function appendMessage(sender, message) {
    const chatWindow = document.getElementById('chat-window');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender);
    messageElement.innerText = message;
    chatWindow.appendChild(messageElement);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function getBotResponse(message) {
    // Placeholder bot response logic
    return "You said: " + message;
}

// Sidebar toggle functionality
const openBtn = document.getElementById('open-sidebar');
const closeBtn = document.getElementById('close-sidebar');
const sidebar = document.getElementById('sidebar');
const chatContainer = document.getElementById('chat-container');
const inputContainer = document.querySelector('.input-container');

openBtn.addEventListener('click', () => {
    sidebar.classList.remove('hidden-sidebar');
    chatContainer.classList.remove('hidden-chat-container');
    inputContainer.classList.remove('hidden-input-container');
});

closeBtn.addEventListener('click', () => {
    sidebar.classList.add('hidden-sidebar');
    chatContainer.classList.add('hidden-chat-container');
    inputContainer.classList.add('hidden-input-container');
});
