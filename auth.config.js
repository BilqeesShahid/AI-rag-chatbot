const { betterAuth } = require("better-auth");

// Configure Better Auth with user model extensions for background collection
const auth = betterAuth({
  database: {
    provider: "sqlite",
    url: process.env.DATABASE_URL || "./db.sqlite",
  },
  // Extend the user model with background fields
  user: {
    schema: {
      // Add software experience field
      software_experience: {
        type: "string",
        required: false,
        defaultValue: "beginner"
      },
      // Add hardware experience field
      hardware_experience: {
        type: "string",
        required: false,
        defaultValue: "beginner"
      }
    }
  },
  // Configure email/password authentication
  emailAndPassword: {
    enabled: true,
    requireEmailVerification: false, // Set to true in production
  },
  // Session management
  session: {
    expiresIn: 7 * 24 * 60 * 60, // 7 days
    rememberMe: true,
  },
  // Account management
  account: {
    accountLinking: {
      enabled: true,
    },
  },
  // OAuth providers (optional, can be added later)
  socialProviders: {
    // google: {
    //   clientId: process.env.GOOGLE_CLIENT_ID,
    //   clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    // },
    // github: {
    //   clientId: process.env.GITHUB_CLIENT_ID,
    //   clientSecret: process.env.GITHUB_CLIENT_SECRET,
    // },
  }
});

module.exports = auth;