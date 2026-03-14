import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';

const firebaseConfig = {
  apiKey: "AIzaSyDamdPWDHKxIfhrtzjF-Nghu8kqqHProrE",
  authDomain: "epiguard-a5b94.firebaseapp.com",
  projectId: "epiguard-a5b94",
  storageBucket: "epiguard-a5b94.firebasestorage.app",
  messagingSenderId: "375799169606",
  appId: "1:375799169606:web:04c7ee5129ca75462e0047"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);