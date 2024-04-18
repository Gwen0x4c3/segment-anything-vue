import { createRouter, createWebHistory } from 'vue-router'
import Segment from '@/views/Segment.vue'

const routes = [
  {
    path: "/",
    name: "Segment",
    component: Segment
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router
