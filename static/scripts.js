// Forecast form submission logic
const form = document.getElementById("forecastForm");
form.addEventListener("submit", async function (event) {
  event.preventDefault();
  const formData = new FormData(form);
  const response = await fetch("/forecast", {
    method: "POST",
    body: formData,
  });
  const result = await response.json();
  if (result.error) {
    document.getElementById("forecastError").innerHTML =
      "<strong>Error: </strong>" + result.error;
  } else {
    document.getElementById("forecastMetrics").innerHTML =
      "<strong>MAE: </strong>" +
      result.mae +
      "<br><strong>R²: </strong>" +
      result.r2;
    const formattedPredictions = result.exact_predictions
      .map((pred) => {
        return `Date: ${new Date(
          pred.ds
        ).toLocaleDateString()}<br>Predicted Orders: ${pred.yhat.toFixed(
          2
        )}`;
      })
      .join("<br><br>");
    document.getElementById("exactPredictions").innerHTML =
      "<strong>Predicted Values: </strong><br>" + formattedPredictions;

    // Parse the plot data
    const plotData = JSON.parse(result.plot).data;

    plotData.forEach((trace, index) => {
      // Setting custom names based on trace content or index
      if (index === 0) trace.name = 'Реални';
      else if (index === 1) trace.name = 'Долна Граница';
      else if (index === 2) trace.name = 'Предсказани';
      else if (index === 3) trace.name = 'Горна Граница';
    });

    // Plotly new plot with dark theme and updated trace colors
    Plotly.newPlot(
      "forecastPlot",
      plotData, // Use updated plot data
      {
        plot_bgcolor: "#FFFFFF",
        paper_bgcolor: "#F1F1F1",
        font: {
          color: "#000000"
        },
        xaxis: {
          tickcolor: "#000000",
          tickfont: {
            color: "#000000"
          },
          titlefont: {
            color: "#000000"
          }
        },
        yaxis: {
          tickcolor: "#000000",
          tickfont: {
            color: "#000000"
          },
          titlefont: {
            color: "#000000"
          }
        }
      }
    );
  }
});

// Particle animation with connections and parallax effect
const canvas = document.getElementById("particles-js");
const ctx = canvas.getContext("2d");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const particles = Array.from({ length: 250 }, () => new Particle());
const maxDistance = 120;
const mouse = { x: null, y: null, radius: 250 };

canvas.addEventListener("mousemove", (event) => {
  mouse.x = event.x;
  mouse.y = event.y;
});

canvas.addEventListener("mouseleave", () => {
  mouse.x = null;
  mouse.y = null;
});

function Particle() {
  this.x = Math.random() * canvas.width;
  this.y = Math.random() * canvas.height;
  this.speedX = (Math.random() - 0.5) * 2;
  this.speedY = (Math.random() - 0.5) * 2;
  this.radius = Math.random() * 3 + 1;
}

Particle.prototype.update = function () {
  this.x += this.speedX;
  this.y += this.speedY;

  if (this.x < 0 || this.x > canvas.width) this.speedX *= -1;
  if (this.y < 0 || this.y > canvas.height) this.speedY *= -1;

  if (mouse.x !== null && mouse.y !== null) {
    const dx = mouse.x - this.x;
    const dy = mouse.y - this.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    if (distance < mouse.radius) {
      const angle = Math.atan2(dy, dx);
      const moveX = Math.cos(angle);
      const moveY = Math.sin(angle);
      this.x -= moveX;
      this.y -= moveY;
    }
  }
};

Particle.prototype.draw = function () {
  ctx.beginPath();
  ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
  ctx.closePath();
  ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
  ctx.fill();
};

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
        ctx.strokeStyle = `rgba(255, 255, 255, ${1 - distance / maxDistance})`;
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }
    }
  }
}

function animateParticles() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  particles.forEach((particle) => {
    particle.update();
    particle.draw();
  });
  connectParticles();
  requestAnimationFrame(animateParticles);
}

animateParticles();
