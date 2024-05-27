const handleSendMessage = async (message) => {
  try {
    const response = await fetch('http://localhost:5000/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json', // Ensure the header is set
      },
      body: JSON.stringify({ msg: message }), // Ensure the body is JSON
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    setMessages([...messages, { user: message, bot: data.response }]);
    setHistory([...history, { user: message, bot: data.response }]);
  } catch (error) {
    console.error('Error sending message:', error);
  }
};
