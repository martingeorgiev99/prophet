// Wait for the DOM to fully load before executing the script
document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("forecastForm");
  const loading = document.getElementById("loading");
  const progressFill = document.querySelector(".progress-fill");

  // Handle form submission
  form.addEventListener("submit", async function (event) {
    event.preventDefault();  // Prevent default form submission behavior

    loading.style.display = "block";  // Show loading indicator
    setTimeout(() => {
      loading.style.opacity = "1";  // Fade in loading indicator
    }, 10);

    const formData = new FormData(form);  // Create FormData object for the form

    progressFill.style.width = "0%";  // Reset progress bar width
    setTimeout(() => {
      progressFill.style.width = "100%";  // Animate progress bar fill
    }, 1000);

    try {
      const response = await fetch("/forecast", {
        method: "POST",  // Send a POST request
        body: formData,  // Attach form data
      });

      loading.style.opacity = "0";  // Fade out loading indicator
      setTimeout(() => {
        loading.style.display = "none";  // Hide loading indicator
        progressFill.style.width = "0";  // Reset progress bar
      }, 500);

      const result = await response.json();  // Parse JSON response

      // Handle errors in the response
      if (result.error) {
        document.getElementById("forecastError").innerHTML = `<strong>Error:</strong> ${result.error}`;
        setTimeout(() => (document.getElementById("forecastError").innerHTML = ""), 5000);  // Clear error after 5 seconds
      } else {
        // Display metrics
        document.getElementById("forecastMetrics").innerHTML = `
          <strong>MAE:</strong> ${result.mae}<br>
          <strong>R²:</strong> ${result.r2}
        `;

        // Format and display exact predictions
        const formattedPredictions = result.exact_predictions
          .map(
            (pred) =>
              `Date: ${new Date(pred.ds).toLocaleDateString()}<br>Predicted Orders: ${pred.yhat.toFixed(2)}`
          )
          .join("<br><br>");
        document.getElementById("exactPredictions").innerHTML = `<strong>Predicted Values: (next 4 weeks)</strong><br>${formattedPredictions}`;

        // Render the plot using Plotly
        Plotly.newPlot(
          "forecastPlot",
          JSON.parse(result.plot).data.map((trace, index) => {
            trace.name =
              ["Реални", "Долна Граница", "Предсказани", "Горна Граница"][index] || trace.name;
            return trace;
          }),
          {
            plot_bgcolor: "#FFFFFF",
            paper_bgcolor: "#F1F1F1",
            font: { color: "#000000" },
            xaxis: {
              tickcolor: "#000000",
              tickfont: { color: "#000000" },
              titlefont: { color: "#000000" },
            },
            yaxis: {
              tickcolor: "#000000",
              tickfont: { color: "#000000" },
              titlefont: { color: "#000000" },
            },
          }
        );
      }
    } catch (error) {
      console.error("Error during fetch:", error);  // Log any fetch errors
    }
  });
});

// Particles and Mouse Parallax
const canvas = document.getElementById("particles-js");
const ctx = canvas.getContext("2d");
canvas.width = window.innerWidth;  // Set canvas width to window width
canvas.height = window.innerHeight;  // Set canvas height to window height

const particles = Array.from({ length: 250 }, () => new Particle());  // Create an array of particles
const maxDistance = 120;  // Maximum distance for particle connection
const mouse = { x: null, y: null, radius: 250 };  // Mouse position tracking

// Update mouse position on mouse move
canvas.addEventListener("mousemove", (event) => {
  mouse.x = event.x;
  mouse.y = event.y;
});

// Reset mouse position when leaving the canvas
canvas.addEventListener("mouseleave", () => {
  mouse.x = null;
  mouse.y = null;
});

// Particle constructor
function Particle() {
  this.x = Math.random() * canvas.width;  // Random x position
  this.y = Math.random() * canvas.height;  // Random y position
  this.speedX = (Math.random() - 0.5) * 2;  // Random x speed
  this.speedY = (Math.random() - 0.5) * 2;  // Random y speed
  this.radius = Math.random() * 3 + 1;  // Random radius
}

// Update particle position
Particle.prototype.update = function () {
  this.x += this.speedX;
  this.y += this.speedY;

  // Reverse direction if particle hits canvas edge
  if (this.x < 0 || this.x > canvas.width) this.speedX *= -1;
  if (this.y < 0 || this.y > canvas.height) this.speedY *= -1;

  // Particle interaction with mouse
  if (mouse.x !== null && mouse.y !== null) {
    const dx = mouse.x - this.x;
    const dy = mouse.y - this.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    if (distance < mouse.radius) {
      const angle = Math.atan2(dy, dx);
      this.x -= Math.cos(angle);  // Move away from mouse
      this.y -= Math.sin(angle);
    }
  }
};

// Draw the particle on canvas
Particle.prototype.draw = function () {
  ctx.beginPath();
  ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
  ctx.closePath();
  ctx.fillStyle = "rgba(255, 255, 255, 0.8)";  // White fill color
  ctx.fill();
};

// Connect particles with lines
function connectParticles() {
  for (let i = 0; i < particles.length; i++) {
    for (let j = i + 1; j < particles.length; j++) {
      const dx = particles[i].x - particles[j].x;
      const dy = particles[i].y - particles[j].y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      if (distance < maxDistance) {
        ctx.beginPath();
        ctx.moveTo(particles[i].x, particles[i].y);
        ctx.lineTo(particles[j].x, particles[j].y);
        ctx.strokeStyle = `rgba(255, 255, 255, ${1 - distance / maxDistance})`;  // Line opacity based on distance
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }
    }
  }
}

// Animate particles on canvas
function animateParticles() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);  // Clear canvas
  particles.forEach((particle) => {
    particle.update();  // Update particle position
    particle.draw();  // Draw particle
  });
  connectParticles();  // Draw connections between particles
  requestAnimationFrame(animateParticles);  // Request next animation frame
}

// Adjust canvas size on window resize
window.addEventListener("resize", () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
});

// Start the particle animation
animateParticles();
