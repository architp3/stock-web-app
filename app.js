const express = require('express');

const app = express();
const { exec } = require('child_process');
const port = 3000;

app.get('/', (req, res) => {
  // Execute your Python script using child_process
  exec('streamlit run app.py', (error, stdout, stderr) => {
    if (error) {
      console.error(`Error: ${error.message}`);
      res.status(500).send('Internal Server Error');
      return;
    }
    console.log(`Python script output: ${stdout}`);
    res.send('Python script executed successfully!');
  });
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
